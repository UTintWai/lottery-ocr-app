import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v69", layout="wide")

def process_v69(img):
    reader = easyocr.Reader(['en'], gpu=False)
    h, w = img.shape[:2]
    target_w = 1600
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ပုံကို ပိုပြတ်သားအောင် လုပ်ခြင်း
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    results = reader.readtext(gray, paragraph=False, mag_ratio=1.5)
    
    # 🎯 FIX: ပုံတစ်ပုံကို ၂၅ တန်းပဲ အသေယူမည်
    grid_rows = [ ["" for _ in range(4)] for _ in range(25) ]
    
    # တန်း ၂၅ တန်းရဲ့ အမြင့်ကို ပိုင်းခြားခြင်း
    row_height = int(gray.shape[0] / 25)
    col_width = int(target_w / 4)
    
    for (bbox, text, prob) in results:
        x_pos = np.mean([p[0] for p in bbox])
        y_pos = np.mean([p[1] for p in bbox])
        
        row_idx = int(y_pos // row_height)
        col_idx = int(x_pos // col_width)
        
        # တန်း ၂၅ တန်းထဲမှာပဲ ရှိနေစေရန်
        if 0 <= row_idx < 25 and 0 <= col_idx < 4:
            txt = text.upper().strip()
            if re.search(r'[။၊"=“_…\.\-\']', txt):
                grid_rows[row_idx][col_idx] = "DITTO"
            else:
                txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if col_idx % 2 == 0: grid_rows[row_idx][col_idx] = num.zfill(3)[-3:]
                    else: grid_rows[row_idx][col_idx] = num
                    
    # Auto-Ditto Filling
    for r in range(25):
        for c in [1, 3]:
            if (grid_rows[r][c] == "DITTO" or grid_rows[r][c] == "") and r > 0:
                grid_rows[r][c] = grid_rows[r-1][c]
                
    return grid_rows

# --- UI ---
st.title("🔢 Lottery Pro v69 (Strict 25-Row Anchor)")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ပုံ (၁)", type=['jpg', 'png', 'jpeg'])
with c2: up_right = st.file_uploader("ပုံ (၂)", type=['jpg', 'png', 'jpeg'])

if st.button("🔍 Process 25-Row Grid"):
    l_data, r_data = [], []
    if up_left: l_data = process_v69(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
    if up_right: r_data = process_v69(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
    
    # ၂၅ တန်းစီ ပေါင်းစပ်ခြင်း
    final = []
    for i in range(25):
        row_l = l_data[i] if l_data else ["","","",""]
        row_r = r_data[i] if r_data else ["","","",""]
        final.append(row_l + row_r)
    st.session_state['v69_data'] = final

if 'v69_data' in st.session_state:
    edited = st.data_editor(st.session_state['v69_data'], num_rows="fixed")
    if st.button("💾 Save"):
        st.success("✅ ၂၅ တန်း စနစ်ဖြင့် သိမ်းဆည်းပြီးပါပြီ!")
