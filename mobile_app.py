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

st.set_page_config(page_title="Lottery Pro 2026 Complete", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- 1. BET LOGIC (SUMMARY အတွက်) ----------------
def process_bet_logic(num_txt, amt_txt):
    """ ဂဏန်းနှင့် ငွေပမာဏကို ခွဲခြမ်းစိတ်ဖြာခြင်း """
    num_clean = re.sub(r'[^0-9R.]', '', str(num_txt).upper())
    amt_clean = re.sub(r'[^0-9]', '', str(amt_txt))
    amt = int(amt_clean) if amt_clean else 0
    results = {}
    
    if 'R' in num_clean:
        base = num_clean.replace('R', '').replace('.', '')
        if len(base) == 3:
            perms = sorted(list(set([''.join(p) for p in permutations(base)])))
            split_amt = amt // len(perms) if len(perms) > 0 else 0
            for p in perms: results[p] = split_amt
        else:
            results[base] = amt
    elif num_clean and num_clean != ".":
        results[num_clean] = amt
    return results

# ---------------- 2. SMART GRID DETECTION ----------------
def get_grid_cells(img_gray, active_cols, num_rows):
    h, w = img_gray.shape
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30))
    
    hor_lines = cv2.dilate(cv2.erode(thresh, horizontal_kernel), horizontal_kernel)
    ver_lines = cv2.dilate(cv2.erode(thresh, vertical_kernel), vertical_kernel)
    
    hor_sum = np.sum(hor_lines, axis=1)
    row_boundaries = np.where(hor_sum > (w * 0.4 * 255))[0]
    
    rows = [0]
    if len(row_boundaries) > 1:
        for i in range(1, len(row_boundaries)):
            if row_boundaries[i] - row_boundaries[i-1] > h // (num_rows * 1.5):
                rows.append(row_boundaries[i])
    if len(rows) <= num_rows: rows = np.linspace(0, h, num_rows + 1).astype(int)
    else: rows.append(h)

    cols = np.linspace(0, w, active_cols + 1).astype(int)
    return rows, cols

# ---------------- 3. MAIN SCANNING ----------------
def scan_voucher(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = get_grid_cells(gray, active_cols, num_rows)
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    
    results = reader.readtext(gray, allowlist='0123456789R.xX')
    
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        c_idx = -1
        for i in range(len(cols)-1):
            if cols[i] <= cx <= cols[i+1]:
                c_idx = i; break
        
        r_idx = -1
        for i in range(len(rows)-1):
            if rows[i] <= cy <= rows[i+1]:
                r_idx = i; break
                
        if r_idx != -1 and c_idx != -1 and r_idx < num_rows:
            grid_data[r_idx][c_idx] = text.replace('x', '*').replace('X', '*')
            
    return grid_data

# ---------------- 4. STREAMLIT UI ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    active_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher တင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("ဖတ်နေပါသည်..."):
            final_data = scan_voucher(img, active_cols, num_rows)
            st.session_state['data_final'] = final_data

# ---------------- 5. EDIT & GOOGLE SHEET ----------------
if 'data_final' in st.session_state:
    st.subheader("📝 အချက်အလက်ပြင်ဆင်ရန်")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            with st.spinner("Sheet ထဲသို့ ပို့ဆောင်နေပါသည်..."):
                # JSON Credentials
                secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
                secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
                creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, 
                        ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
                client = gspread.authorize(creds)
                
                ss = client.open("LotteryData")
                sh1 = ss.get_worksheet(0) # Sheet1
                sh2 = ss.get_worksheet(1) # Sheet2 (Summary)
                
                # Sheet 1 သို့ မူရင်းဒေတာပို့ခြင်း
                sh1.append_rows(edited_data)

                # Summary တွက်ချက်ခြင်း
                master_sum = {}
                for row in edited_data:
                    for i in range(0, len(row)-1, 2):
                        if row[i] and row[i+1]:
                            res = process_bet_logic(row[i], row[i+1])
                            for k, v in res.items():
                                master_sum[k] = master_sum.get(k, 0) + v

                # Sheet 2 သို့ Summary ပို့ခြင်း
                sh2.clear()
                summary_list = [[k, v] for k, v in sorted(master_sum.items())]
                sh2.append_rows([["Number", "Total"]] + summary_list)
                
                st.success("✅ Google Sheet သို့ အောင်မြင်စွာ ပို့ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"Error: {e}")
