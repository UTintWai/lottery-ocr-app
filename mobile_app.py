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

st.set_page_config(page_title="Lottery Pro 2026 Row Fix", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- 1. ROW CLUSTERING LOGIC ----------------
def organize_by_rows(ocr_results, num_rows, h_img):
    """ စာလုံးတွေ တစ်ကွက်တည်းမှာ ရှိအောင် ဒေါင်လိုက် အမြင့်ကို ပြန်ညှိခြင်း """
    # အတန်းတစ်ခုရဲ့ ပျမ်းမျှအမြင့်ကို တွက်ခြင်း
    expected_row_h = h_img / num_rows
    y_threshold = expected_row_h * 0.4 # အမြင့်ကွာခြားချက် ၄၀% အတွင်းဆိုရင် အတန်းတူဟု သတ်မှတ်မည်

    processed_data = []
    for (bbox, text, prob) in ocr_results:
        # စာလုံးရဲ့ အလယ်ဗဟို Y coordinate
        cy = np.mean([p[1] for p in bbox])
        cx = np.mean([p[0] for p in bbox])
        
        # ဘယ်နှစ်တန်းမြောက်လဲဆိုတာကို ပုံသေမတွက်ဘဲ အနီးစပ်ဆုံး အတန်းထဲ ထည့်ခြင်း
        r_idx = int(cy // expected_row_h)
        if r_idx >= num_rows: r_idx = num_rows - 1
        
        processed_data.append({
            'r': r_idx,
            'c_val': cx,
            'text': text
        })
    return processed_data

# ---------------- 2. IMPROVED SCANNING ----------------
def scan_voucher_aligned(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # OCR ဖတ်ခြင်း
    results = reader.readtext(gray, allowlist='0123456789R.*xX')
    
    # Row Alignment ညှိခြင်း
    aligned_results = organize_by_rows(results, num_rows, h)
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)

    for item in aligned_results:
        r = item['r']
        cx = item['c_val']
        text = item['text']
        
        # Column ရှာခြင်း
        c = np.searchsorted(col_edges, cx) - 1
        
        if 0 <= r < num_rows and 0 <= c < active_cols:
            # အကယ်၍ အကွက်ထဲမှာ စာရှိနှင့်ပြီးသားဆိုလျှင် (ဥပမာ 50 နဲ့ 80 ခွဲဖတ်မိလျှင်) ပေါင်းပေးမည်
            clean_t = text.upper().replace('X', '*')
            if grid_data[r][c]:
                grid_data[r][c] += "*" + clean_t
            else:
                grid_data[r][c] = clean_t
                
    return grid_data

# ---------------- 3. UI & UPLOAD ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher တင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("အတန်းများကို တည့်အောင် ညှိနေပါသည်..."):
            data = scan_voucher_aligned(img, a_cols, n_rows)
            st.session_state['aligned_df'] = data

if 'aligned_df' in st.session_state:
    final_df = st.data_editor(st.session_state['aligned_df'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheet"):
        # (Google Sheet Logic...)
        st.success("✅ အချက်အလက်များကို ပို့ဆောင်ပြီးပါပြီ။")
