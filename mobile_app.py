import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- BET LOGIC ----------------
def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3:
        return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X','*').replace('×','*')
    results = {}

    if 'R' in clean_num:
        base = clean_num.replace('R','')
        perms = get_all_permutations(base)
        amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
        if perms and amt>0:
            split = amt // len(perms)
            for p in perms:
                results[p]=split

    elif '*' in amt_str:
        parts = amt_str.split('*')
        if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():
            base_amt = int(parts[0])
            total_amt = int(parts[1])
            num_final = clean_num.zfill(3)
            results[num_final] = base_amt
            perms = [p for p in get_all_permutations(num_final) if p!=num_final]
            if perms:
                split = (total_amt-base_amt)//len(perms)
                for p in perms:
                    results[p]=split
    else:
        amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
        num_final = clean_num.zfill(3)
        if num_final:
            results[num_final]=amt

    return results

# ---------------- SETTINGS ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = 25
    num_cols_active = 8
    st.success("Grid Locked: 25 x 8")

st.title("🎰 Lottery OCR PRO (Cell Accurate)")

uploaded_file = st.file_uploader("📥 ပုံတင်ရန်", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):
        with st.spinner("Scanning..."):

            h, w = gray.shape
            col_width = w / num_cols_active
            row_height = h / num_rows

            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

            for r in range(num_rows):
                for c in range(num_cols_active):

                    x1 = int(c * col_width)
                    x2 = int((c+1) * col_width)
                    y1 = int(r * row_height)
                    y2 = int((r+1) * row_height)

                    cell = gray[y1:y2, x1:x2]

                    # enlarge
                    cell = cv2.resize(cell, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # blur
                    cell = cv2.GaussianBlur(cell,(3,3),0)

                    # adaptive threshold
                    cell = cv2.adaptiveThreshold(
                        cell,255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,
                        11,2)

                    result = reader.readtext(cell, detail=0, paragraph=False)

                    txt = result[0] if result else ""
                    txt = txt.upper().strip()

                    # OCR correction
                    txt = txt.replace('S','5')
                    txt = txt.replace('I','1')
                    txt = txt.replace('Z','7')
                    txt = txt.replace('G','6')
                    txt = txt.replace('O','0')
                    txt = txt.replace('T','1')
                    txt = txt.replace('X','*')
                    txt = txt.replace('×','*')

                    grid_data[r][c] = txt

            # ---------- CLEANING ----------
            for c in range(num_cols_active):
                last_val=""
                for r in range(num_rows):
                    curr=str(grid_data[r][c]).strip()

                    if c%2==0:
                        curr=re.sub(r'[^0-9R]','',curr)
                        if curr=="" and last_val:
                            grid_data[r][c]=last_val
                        else:
                            if curr.isdigit():
                                curr=curr[-3:].zfill(3)
                            grid_data[r][c]=curr
                            if curr:
                                last_val=curr
                    else:
                        nums=re.findall(r'\d+',curr)
                        grid_data[r][c]=max(nums,key=lambda x:int(x)) if nums else ""

            st.session_state['data_final']=grid_data

# ---------------- UPLOAD ----------------
if 'data_final' in st.session_state:

    st.subheader("📝 စစ်ဆေးပြီး Upload")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Google Sheet Upload"):
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n","\n")

            scope = [
                "https://spreadsheets.google.com/feeds",
                "https://www.googleapis.com/auth/drive"
            ]

            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)

            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            sh2 = ss.get_worksheet(1)

            sh1.append_rows(edited_data)

            master_sum={}
            for row in edited_data:
                for i in range(0,8,2):
                    n_txt=str(row[i]).strip()
                    a_txt=str(row[i+1]).strip()
                    if n_txt and a_txt:
                        bet_res=process_bet_logic(n_txt,a_txt)
                        for g,val in bet_res.items():
                            master_sum[g]=master_sum.get(g,0)+val

            sh2.clear()
            final_list=[[k,master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["ဂဏန်း","စုစုပေါင်း"]]+final_list)

            st.success("🎉 Upload Complete!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
