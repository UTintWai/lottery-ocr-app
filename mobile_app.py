import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import os
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Lottery Pro 2026 Precise", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- 1. PROCESSING LOGIC ----------------
def process_bet_logic(num_txt, amt_txt):
    num = re.sub(r'[^0-9R]', '', str(num_txt))
    amt_str = re.sub(r'[^0-9]', '', str(amt_txt))
    amt = int(amt_str) if amt_str else 0
    results = {}
    if 'R' in num:
        base = num.replace('R', '')
        if len(base) == 3:
            perms = sorted(list(set([''.join(p) for p in permutations(base)])))
            for p in perms: results[p] = amt // len(perms) if len(perms) > 0 else 0
        elif len(base) == 2:
            results[base] = amt; results[base[::-1]] = amt
    elif num:
        results[num] = amt
    return results

# ---------------- 2. CELL-BY-CELL SCANNING ----------------
def scan_voucher_by_cell(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # ဇယားကွက်အနားသတ်များကို တွက်ချက်ခြင်း
    row_edges = np.linspace(0, h, num_rows + 1).astype(int)
    col_edges = np.linspace(0, w, active_cols + 1).astype(int)
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    
    progress_bar = st.progress(0)
    total_cells = num_rows * active_cols
    current_cell = 0

    for r in range(num_rows):
        for c in range(active_cols):
            # အကွက်တစ်ခုချင်းစီကို ပုံဖြတ်ယူခြင်း (Cell Cropping)
            y1, y2 = row_edges[r], row_edges[r+1]
            x1, x2 = col_edges[c], col_edges[c+1]
            
            # အကွက်ရဲ့ အလယ်ဗဟိုနားကပုံကိုပဲ ယူမယ် (ဘေးမျဉ်းကြောင်းတွေ မပါအောင်)
            cell_roi = gray[y1+2:y2-2, x1+2:x2-2]
            
            # ဖြတ်ထားတဲ့ အကွက်သေးလေးကိုပဲ OCR ဖတ်မယ်
            result = reader.readtext(cell_roi, allowlist='0123456789R.*')
            
            if result:
                # အကွက်ထဲမှာ စာသားတွေ့ရင် အမှားတွေကို သန့်စင်မယ်
                raw_text = "".join([res[1] for res in result])
                clean_text = re.sub(r'[^0-9R.*]', '', raw_text)
                grid_data[r][c] = clean_text
            
            current_cell += 1
            progress_bar.progress(current_cell / total_cells)
            
    return grid_data

# ---------------- 3. UI ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    active_cols = st.selectbox("အတိုင်အရေအတွက် (Column)", [2, 4, 6, 8], index=2)
    num_rows = st.number_input("အတန်းအရေအတွက် (Row)", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("အကွက်တစ်ခုချင်းစီကို အသေးစိတ် ဖတ်နေပါသည်..."):
            final_data = scan_voucher_by_cell(img, active_cols, num_rows)
            st.session_state['precise_data'] = final_data

if 'precise_data' in st.session_state:
    edited_df = st.data_editor(st.session_state['precise_data'], use_container_width=True)

    if st.button("🚀 Send to Google Sheet"):
        # (Google Sheet upload code as provided before)
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_df)
            
            # Summary Table Logic
            sh2 = ss.get_worksheet(1)
            master_summary = {}
            for row in edited_df:
                for i in range(0, len(row)-1, 2):
                    if row[i] and row[i+1]:
                        res = process_bet_logic(row[i], row[i+1])
                        for k, v in res.items():
                            master_summary[k] = master_summary.get(k, 0) + v
            
            sh2.clear()
            summary_list = [[k, v] for k, v in sorted(master_summary.items())]
            sh2.append_rows([["Number", "Amount"]] + summary_list)
            st.success("✅ Google Sheet သို့ ပို့ပြီးပါပြီ။")
        except Exception as e:
            st.error(f"Error: {e}")
