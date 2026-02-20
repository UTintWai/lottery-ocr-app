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

st.set_page_config(page_title="Lottery Pro 2026 Smart Grid", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- 1. SMART GRID DETECTION ----------------
def get_grid_cells(img_gray, active_cols, num_rows):
    """ Voucher ထဲက ဇယားကွက်တွေကို တိတိကျကျ ပိုင်းဖြတ်ပေးမည့် logic """
    h, w = img_gray.shape
    
    # မျဉ်းကြောင်းများကို ပေါ်လွင်အောင်လုပ်ခြင်း
    thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # အလျားလိုက်နှင့် ဒေါင်လိုက် မျဉ်းများကို ရှာဖွေခြင်း
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 30, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 30))
    
    hor_lines = cv2.erode(thresh, horizontal_kernel, iterations=1)
    hor_lines = cv2.dilate(hor_lines, horizontal_kernel, iterations=1)
    
    ver_lines = cv2.erode(thresh, vertical_kernel, iterations=1)
    ver_lines = cv2.dilate(ver_lines, vertical_kernel, iterations=1)
    
    # မျဉ်းများ မတွေ့ပါက ပုံသေ Grid စနစ်သို့ ပြန်ပြောင်းသုံးမည်
    hor_sum = np.sum(hor_lines, axis=1)
    ver_sum = np.sum(ver_lines, axis=0)
    
    row_boundaries = np.where(hor_sum > (w * 0.5 * 255))[0]
    col_boundaries = np.where(ver_sum > (h * 0.5 * 255))[0]
    
    # မျဉ်းကြောင်းများ တိကျစွာ မတွေ့ပါက (စက္ကူမှာ မျဉ်းမပါလျှင်) ပုံသေ အချိုးချမည်
    rows = []
    if len(row_boundaries) < 2:
        rows = np.linspace(0, h, num_rows + 1).astype(int)
    else:
        # မျဉ်းကြောင်းများအကြား အကွာအဝေးကို တွက်ချက်ခြင်း
        rows = [row_boundaries[0]]
        for i in range(1, len(row_boundaries)):
            if row_boundaries[i] - rows[-1] > h // (num_rows * 2):
                rows.append(row_boundaries[i])
        if len(rows) <= num_rows: rows = np.linspace(0, h, num_rows + 1).astype(int)

    cols = np.linspace(0, w, active_cols + 1).astype(int)
    
    return rows, cols

# ---------------- 2. MAIN LOGIC ----------------
def scan_voucher(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rows, cols = get_grid_cells(gray, active_cols, num_rows)
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    
    # OCR ကို တစ်ကွက်ချင်းစီထက် အုပ်စုလိုက် ဖတ်ခိုင်းခြင်းက ပိုမြန်ပြီး တိကျသည်
    results = reader.readtext(gray, allowlist='0123456789R.xX')
    
    for (bbox, text, prob) in results:
        # စာလုံး၏ ဗဟိုချက်ကို ရှာသည်
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        # မည်သည့် အကွက်ထဲတွင် ရှိနေသလဲ ရှာဖွေခြင်း
        c_idx = -1
        for i in range(len(cols)-1):
            if cols[i] <= cx <= cols[i+1]:
                c_idx = i
                break
        
        r_idx = -1
        for i in range(len(rows)-1):
            if rows[i] <= cy <= rows[i+1]:
                r_idx = i
                break
                
        if r_idx != -1 and c_idx != -1 and r_idx < num_rows:
            # ဂဏန်းများ သန့်စင်ခြင်း (ဥပမာ x ကို * ပြောင်းခြင်း)
            clean_text = text.replace('x', '*').replace('X', '*')
            grid_data[r_idx][c_idx] = clean_text
            
    return grid_data

# ---------------- 3. STREAMLIT UI ----------------
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
        with st.spinner("ဇယားကွက်များနှင့် ဂဏန်းများကို တိုက်စစ်နေသည်..."):
            final_data = scan_voucher(img, active_cols, num_rows)
            st.session_state['data_final'] = final_data
            st.success("Scanning ပြီးပါပြီ!")

if 'data_final' in st.session_state:
    st.data_editor(st.session_state['data_final'], use_container_width=True)
