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

# --- OCR & LOGIC ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

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

# --- MAIN SCAN FUNCTION ---
def scan_voucher_final(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    results = reader.readtext(gray, allowlist='0123456789R.*xX')
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        c = np.searchsorted(col_edges, cx) - 1
        r = np.searchsorted(row_edges, cy) - 1
        
        if 0 <= r < num_rows and 0 <= c < active_cols:
            t = text.upper().replace('X', '*')
            grid_data[r][c] = t
    return grid_data

# --- UI ---
st.title("Lottery Pro 2026 (Sheet Fix)")

with st.sidebar:
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher တင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        data = scan_voucher_final(img, a_cols, n_rows)
        st.session_state['sheet_data'] = data

if 'sheet_data' in st.session_state:
    edited_df = st.data_editor(st.session_state['sheet_data'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheet"):
        try:
            # ၁။ Secrets ထဲက JSON ကို ဖတ်ခြင်း
            if "GCP_SERVICE_ACCOUNT_FILE" not in st.secrets:
                st.error("Secrets ထဲမှာ GCP_SERVICE_ACCOUNT_FILE မရှိပါဘူးဗျ။")
                st.stop()
                
            info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            info["private_key"] = info["private_key"].replace("\\n", "\n")
            
            # ၂။ Authenticate လုပ်ခြင်း
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
            client = gspread.authorize(creds)
            
            # ၃။ Sheet ကို ဖွင့်ခြင်း
            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            sh2 = ss.get_worksheet(1)
            
            # ၄။ Data ပို့ခြင်း
            with st.spinner("ပို့ဆောင်နေပါသည်..."):
                # ဗလာဖြစ်နေတဲ့ row တွေကို ဖယ်ထုတ်ပါမယ်
                clean_rows = [row for row in edited_df if any(row)]
                if clean_rows:
                    sh1.append_rows(clean_rows)
                
                # Summary တွက်ချက်ခြင်း
                master_sum = {}
                for row in edited_df:
                    for i in range(0, len(row)-1, 2):
                        if row[i] and row[i+1]:
                            res = process_bet_logic(row[i], row[i+1])
                            for k, v in res.items():
                                master_sum[k] = master_sum.get(k, 0) + v
                
                if master_sum:
                    sh2.clear()
                    summary_list = [[k, v] for k, v in sorted(master_sum.items())]
                    sh2.append_rows([["Number", "Amount"]] + summary_list)
                
                st.success("✅ Sheet ထဲကို ရောက်သွားပါပြီဗျ!")
                
        except gspread.exceptions.SpreadsheetNotFound:
            st.error("Error: 'LotteryData' ဆိုတဲ့ Sheet နာမည်ကို ရှာမတွေ့ပါဘူး။")
        except Exception as e:
            st.error(f"Error: {e}")
