import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- ၁။ CONFIG & OCR LOAD ----------------
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3: return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X','*')
    results = {}
    try:
        if 'R' in clean_num:
            base = clean_num.replace('R','')
            perms = get_all_permutations(base)
            amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
            if perms and amt > 0:
                split = amt // len(perms)
                for p in perms: results[p] = split
        elif '*' in amt_str:
            parts = amt_str.split('*')
            if len(parts)==2:
                base_amt, total_amt = int(parts[0]), int(parts[1])
                num_final = clean_num.zfill(3)
                results[num_final] = base_amt
                perms = [p for p in get_all_permutations(num_final) if p!=num_final]
                if perms:
                    split = (total_amt-base_amt)//len(perms)
                    for p in perms: results[p] = split
        else:
            amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
            num_final = clean_num.zfill(3) if (clean_num.isdigit() and len(clean_num)<=3) else clean_num
            if num_final: results[num_final] = amt
    except: pass
    return results

# ---------------- ၂။ SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("Rows", min_value=1, value=25)
    col_mode = st.selectbox("Columns", ["2","4","6","8"], index=2) # Default to 6
    num_cols_active = int(col_mode)

# ---------------- ၃။ OCR SCAN LOGIC ----------------
st.title("🎰 Lottery OCR Stable Version")
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 စစ်ဆေးမည် (OCR Scan)"):
        with st.spinner(f"{num_cols_active} တိုင်ကို ဖတ်နေပါသည်..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
            processed_img = clahe.apply(gray)
            
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]
            results = reader.readtext(processed_img, detail=1, contrast_ths=0.05)

            # ၆ တိုင်စနစ်အတွက် D1-6 မိစေရန် Column Boundary ညှိခြင်း
            if num_cols_active == 6:
                col_steps = [0.18, 0.35, 0.52, 0.68, 0.85, 1.0]
            else:
                col_steps = [(i+1)/num_cols_active for i in range(num_cols_active)]

            for (bbox, text, prob) in results:
                left_x = bbox[0][0]
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                rel_x = (left_x * 0.3 + cx * 0.7) / w
                c_idx = next((i for i, s in enumerate(col_steps) if rel_x <= s), num_cols_active-1)
                r_idx = int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:
                    txt = text.upper().strip().replace('S','5').replace('T','7').replace('Z','7').replace('G','6').replace('O','0').replace('I','1')
                    if c_idx % 2 == 0: # Number Column
                        txt = re.sub(r'[^0-9R]', '', txt)
                        if len(txt) == 2 and txt.isdigit(): txt = "0" + txt
                        elif len(txt) > 3 and 'R' not in txt: txt = txt[:3]
                    else: # Amount Column
                        txt = re.sub(r'[^0-9X*]', '', txt)
                    grid_data[r_idx][c_idx] = txt

            # Ditto Logic
            for c in range(num_cols_active):
                last_v = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if curr in ['"', "''", "4", "LL", "V", "11", "U", "-"] and last_v: grid_data[r][c] = last_v
                    elif curr: last_v = curr
            st.session_state['data_final'] = grid_data

# ---------------- ၄။ GOOGLE SHEET EXPORT ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Upload to Google Sheet"):
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n","\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw Data
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_data)

            # Sheet 2: Summing Logic
            master_sum = {}
            for row in edited_data:
                # Dynamic range based on active columns
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        bet_res = process_bet_logic(n, a)
                        for k, v in bet_res.items(): master_sum[k] = master_sum.get(k, 0) + v

            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["Number", "Total"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.success("✅ Sheet 1 & 2 သို့ ဒေတာများ ပို့ဆောင်ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")