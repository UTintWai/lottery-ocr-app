import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Lottery Pro 2026 Stable", layout="wide")

# ---------------- OCR LOADER ----------------
@st.cache_resource
def load_ocr():
    # GPU မရှိတဲ့ Cloud ပေါ်မှာ run ရင် gpu=False ထားရပါမယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- FUNCTIONS ----------------
def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', str(num_str))
    if len(num_only) != 3:
        return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X', '*')
    results = {}
    try:
        if 'R' in clean_num:
            base = clean_num.replace('R', '')
            perms = get_all_permutations(base)
            # Find all numbers in amount string
            match_amt = re.findall(r'\d+', amt_str)
            amt = int(match_amt[0]) if match_amt else 0
            if perms and amt > 0:
                split = amt // len(perms)
                for p in perms: results[p] = split
        elif '*' in amt_str:
            parts = amt_str.split('*')
            if len(parts) == 2:
                base_amt = int(re.sub(r'\D', '', parts[0]))
                total_amt = int(re.sub(r'\D', '', parts[1]))
                num_final = clean_num.zfill(3)
                results[num_final] = base_amt
                perms = [p for p in get_all_permutations(num_final) if p != num_final]
                if perms:
                    split = (total_amt - base_amt) // len(perms)
                    for p in perms: results[p] = split
        else:
            match_amt = re.findall(r'\d+', amt_str)
            amt = int(match_amt[0]) if match_amt else 0
            num_final = clean_num.zfill(3) if clean_num.isdigit() else clean_num
            if num_final: results[num_final] = amt
    except: pass
    return results

def clean_ocr_text(txt):
    txt = txt.upper().strip()
    repls = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'7','T':'7','Q':'0','D':'0'}
    for k,v in repls.items(): txt = txt.replace(k,v)
    return txt

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("Rows", min_value=1, value=25)
    col_mode = st.selectbox("Columns", ["2","4","6","8"], index=3)

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload Voucher Image", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):
        with st.spinner("Processing..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Contrast တိုးမြှင့်ခြင်း
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
            h, w = processed.shape

            num_cols_active = int(col_mode)
            col_width = w / num_cols_active
            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

            # OCR Scanning
            results = reader.readtext(processed)

            for (bbox, text, prob) in results:
                if prob < 0.25: continue
                
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                c_idx = int(cx / col_width)
                r_idx = int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:
                    clean_txt = clean_ocr_text(text)
                    nums = re.findall(r'\d+', clean_txt)
                    
                    if c_idx % 2 == 0: # Number Column
                        grid_data[r_idx][c_idx] = nums[0].zfill(3) if nums else ""
                    else: # Amount Column
                        grid_data[r_idx][c_idx] = max(nums, key=int) if nums else ""

            # Ditto Logic
            for c in range(num_cols_active):
                last_val = ""
                for r in range(num_rows):
                    if grid_data[r][c] == "":
                        grid_data[r][c] = last_val
                    else:
                        last_val = grid_data[r][c]

            st.session_state['data_final'] = grid_data
            st.success("Scan Complete!")

# ---------------- GOOGLE SHEET ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Upload to Google Sheet"):
        try:
            if "GCP_SERVICE_ACCOUNT_FILE" not in st.secrets:
                st.error("Secrets setup မလုပ်ရသေးပါခင်ဗျာ!")
            else:
                secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
                secret_info["private_key"] = secret_info["private_key"].replace("\\n","\n")
                scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
                creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
                client = gspread.authorize(creds)

                ss = client.open("LotteryData")
                sh1 = ss.get_worksheet(0)
                sh2 = ss.get_worksheet(1)

                sh1.append_rows(edited_data)

                # Process Summary
                master_sum = {}
                for row in edited_data:
                    for i in range(0, len(row)-1, 2):
                        n, a = str(row[i]), str(row[i+1])
                        if n and a:
                            bet_res = process_bet_logic(n, a)
                            for k, v in bet_res.items():
                                master_sum[k] = master_sum.get(k, 0) + v

                sh2.clear()
                final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
                sh2.append_rows([["Number","Total"]] + final_list)
                st.success("✅ Uploaded Successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
