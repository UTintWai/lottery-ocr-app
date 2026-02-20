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

st.set_page_config(page_title="Lottery Pro 2026 Precision", layout="wide")

@st.cache_resource
def load_ocr():
    # အက္ခရာတွေကို သီးသန့် ကန့်သတ်ထားခြင်းဖြင့် အမှားနည်းစေပါသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- 1. DATA CLEANING LOGIC ----------------
def clean_digit_text(text):
    """ ဖတ်လိုက်တဲ့ စာသားထဲက အမှားတွေကို ပြန်ပြင်ခြင်း """
    # x ကို * ပြောင်းခြင်း၊ မလိုအပ်တဲ့ စာလုံးများ ဖယ်ခြင်း
    t = text.upper().replace('X', '*').replace('x', '*')
    # ဂဏန်း၊ R နှင့် * ၊ . မှလွဲ၍ အကုန်ဖယ်သည်
    t = re.sub(r'[^0-9R.*]', '', t)
    
    # အဖြစ်များသော အမှားများ ပြင်ခြင်း (ဥပမာ 1 ကို 7 ဖြစ်နိုင်ခြေရှိလျှင် template matching ပိုသုံးသင့်သည်)
    if t == 'S': t = '5'
    if t == 'B': t = '8'
    return t

def process_bet_logic(num_txt, amt_txt):
    """ ဂဏန်းပေါင်းခြင်းနှင့် R လှည့်ခြင်း """
    num = re.sub(r'[^0-9R]', '', str(num_txt))
    amt_str = re.sub(r'[^0-9]', '', str(amt_txt))
    amt = int(amt_str) if amt_str else 0
    results = {}

    if 'R' in num:
        base = num.replace('R', '')
        if len(base) == 3:
            perms = sorted(list(set([''.join(p) for p in permutations(base)])))
            for p in perms: results[p] = amt // len(perms)
        elif len(base) == 2:
            results[base] = amt; results[base[::-1]] = amt
    elif num:
        results[num] = amt
    return results

# ---------------- 2. SMART COORDINATE LOGIC ----------------
def scan_voucher_precise(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # OCR ဖတ်ခြင်း (Paragraph mode ကို ပိတ်ထားခြင်းက အကွက်လွဲမှုကို ကာကွယ်သည်)
    results = reader.readtext(gray, allowlist='0123456789R.*')
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    
    # ဒေါင်လိုက်နှင့် အလျားလိုက် အကွက်စိတ်များကို ပိုမိုတိကျစွာ တွက်ခြင်း
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    for (bbox, text, prob) in results:
        # ဗဟိုချက်ကို တွက်ချက်ခြင်း
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        # Column ရှာခြင်း
        c_idx = np.searchsorted(col_edges, cx) - 1
        # Row ရှာခြင်း
        r_idx = np.searchsorted(row_edges, cy) - 1
        
        if 0 <= r_idx < num_rows and 0 <= c_idx < active_cols:
            cleaned = clean_digit_text(text)
            # အကွက်တစ်ခုတည်းမှာ စာသားထပ်နေရင် ပေါင်းပေးမည် (သို့မဟုတ်) အသစ်ဖြင့် လဲမည်
            grid_data[r_idx][c_idx] = cleaned

    return grid_data

# ---------------- 3. UI & GOOGLE SHEET ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    cols_count = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    rows_count = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher တင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("အမှားနည်းအောင် သေချာဖတ်နေပါသည်..."):
            scanned_data = scan_voucher_precise(img, cols_count, rows_count)
            st.session_state['final_table'] = scanned_data

if 'final_table' in st.session_state:
    st.info("💡 အမှားပါက အကွက်ထဲတွင် တိုက်ရိုက်ပြင်နိုင်ပါသည်")
    edited_df = st.data_editor(st.session_state['final_table'], use_container_width=True)

    if st.button("🚀 Send to Google Sheet"):
        try:
            # Google Sheet Connection
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, 
                        ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Original Data
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_df)
            
            # Sheet 2: Summary Logic
            sh2 = ss.get_worksheet(1)
            master_summary = {}
            for row in edited_df:
                for i in range(0, len(row)-1, 2):
                    if row[i] and row[i+1]:
                        res = process_bet_logic(row[i], row[i+1])
                        for k, v in res.items():
                            master_summary[k] = master_summary.get(k, 0) + v
            
            sh2.clear()
            final_rows = [[k, v] for k, v in sorted(master_summary.items())]
            sh2.append_rows([["Number", "Amount"]] + final_rows)
            
            st.success("✅ အချက်အလက်များကို ပို့ဆောင်ပြီးပါပြီ။")
        except Exception as e:
            st.error(f"Error: {e}")
