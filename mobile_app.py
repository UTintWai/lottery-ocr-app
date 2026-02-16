import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery PRO 2026", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- SETTINGS ----------------
NUM_ROWS = 25
NUM_COLS = 8

st.title("🎰 Lottery OCR PRO (Cell-by-Cell Accurate Version)")
uploaded_file = st.file_uploader("📥 လက်ရေးမူပုံတင်ရန်", type=["jpg","jpeg","png"])


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
        base_num = clean_num.replace('R', '')
        perms = get_all_permutations(base_num)
        amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
        if perms and amt > 0:
            split_amt = amt // len(perms)
            for p in perms:
                results[p] = split_amt

    elif '*' in amt_str:
        parts = amt_str.split('*')
        if len(parts)==2 and parts[0].isdigit() and parts[1].isdigit():
            base_amt = int(parts[0])
            total_amt = int(parts[1])
            num_final = clean_num.zfill(3)
            results[num_final] = base_amt
            perms = [p for p in get_all_permutations(num_final) if p != num_final]
            if perms:
                split_amt = (total_amt - base_amt)//len(perms)
                for p in perms:
                    results[p] = split_amt
    else:
        amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
        num_final = clean_num.zfill(3)
        if num_final:
            results[num_final] = amt

    return results


# ---------------- OCR PROCESS ----------------
if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):

        with st.spinner("Scanning..."):

            h, w = img.shape[:2]
            col_w = w / NUM_COLS
            row_h = h / NUM_ROWS

            # Strong preprocessing
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            grid_data = [["" for _ in range(NUM_COLS)] for _ in range(NUM_ROWS)]

            for r in range(NUM_ROWS):
                for c in range(NUM_COLS):

                    x1 = int(c * col_w)
                    x2 = int((c+1) * col_w)
                    y1 = int(r * row_h)
                    y2 = int((r+1) * row_h)

                    cell = gray[y1:y2, x1:x2]

                    # enlarge for better OCR
                    cell = cv2.resize(cell, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

                    # adaptive threshold
                    cell_thresh = cv2.adaptiveThreshold(
                        cell,255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY,11,2)

                    # OCR pass 1
                    result1 = reader.readtext(cell_thresh, detail=0)

                    # OCR pass 2 (invert)
                    cell_inv = cv2.bitwise_not(cell_thresh)
                    result2 = reader.readtext(cell_inv, detail=0)

                    txt = ""

                    if result1:
                        txt = result1[0]
                    elif result2:
                        txt = result2[0]

                    txt = txt.upper().strip()

                    # OCR corrections
                    txt = txt.replace('S','5')
                    txt = txt.replace('I','1')
                    txt = txt.replace('Z','7')
                    txt = txt.replace('G','6')
                    txt = txt.replace('O','0')
                    txt = txt.replace('T','1')
                    txt = txt.replace('X','*')
                    txt = txt.replace('×','*')

                    txt = re.sub(r'\.+','.',txt)

                    grid_data[r][c] = txt

            # -------- CLEANING PHASE --------
            for c in range(NUM_COLS):
                last_val = ""
                for r in range(NUM_ROWS):

                    curr = grid_data[r][c].strip()

                    if c % 2 == 0:
                        # Number column
                        curr = re.sub(r'[^0-9R]', '', curr)

                        if curr == "" and last_val:
                            grid_data[r][c] = last_val
                        else:
                            if curr.isdigit():
                                curr = curr[-3:].zfill(3)
                            grid_data[r][c] = curr
                            if curr:
                                last_val = curr
                    else:
                        # Amount column
                        nums = re.findall(r'\d+', curr)
                        grid_data[r][c] = max(nums, key=lambda x: int(x)) if nums else ""

            st.session_state["data_final"] = grid_data
            st.success("✅ OCR Completed (200 Cells Processed)")


# ---------------- GOOGLE SHEET ----------------
if "data_final" in st.session_state:

    st.subheader("📝 Edit & Upload")
    edited_data = st.data_editor(st.session_state["data_final"], use_container_width=True)

    if st.button("🚀 Upload to Google Sheet"):

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

            master_sum = {}

            for row in edited_data:
                for i in range(0, NUM_COLS, 2):
                    n_txt = str(row[i]).strip()
                    a_txt = str(row[i+1]).strip()
                    if n_txt and a_txt:
                        bet_res = process_bet_logic(n_txt, a_txt)
                        for g,val in bet_res.items():
                            master_sum[g] = master_sum.get(g,0)+val

            sh2.clear()
            final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["ဂဏန်း","စုစုပေါင်း"]] + final_list)

            st.success("🎉 Upload Complete!")

        except Exception as e:
            st.error(f"❌ Error: {e}")
