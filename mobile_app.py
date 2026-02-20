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
st.set_page_config(page_title="Lottery Pro 2026 Stable OCR", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_digit_templates():
    templates = {}
    temp_path = "templates" 
    if os.path.exists(temp_path):
        for i in range(10):
            p = os.path.join(temp_path, f"{i}.png")
            if not os.path.exists(p): p = os.path.join(temp_path, f"{i}.jpg")
            if os.path.exists(p):
                img = cv2.imread(p, 0)
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                templates[str(i)] = cv2.resize(img, (28, 28))
    return templates

reader = load_ocr()
digit_templates = load_digit_templates()

# ---------------- HELPERS ----------------
def match_digit_from_image(roi_gray):
    if not digit_templates: return None
    roi = cv2.resize(roi_gray, (28, 28))
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    best_score = -1
    best_digit = None
    for digit, temp_img in digit_templates.items():
        res = cv2.matchTemplate(roi, temp_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_digit = digit
    return best_digit if best_score > 0.45 else None # Score ကို နည်းနည်းလျှော့ထားပါတယ်

def clean_ocr_text(txt):
    txt = txt.upper().strip()
    # အမှားများတဲ့ စာလုံးတွေကို ဂဏန်းပြောင်းခြင်း
    repls = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'7','T':'7','Q':'0','D':'0'}
    for k,v in repls.items(): txt = txt.replace(k,v)
    return txt

# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload Voucher", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 Full Scan & Match"):
        with st.spinner("တစ်ကွက်မကျန် ဖတ်နေသည်..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # အရင် code လိုပဲ ပုံကို ကြည်အောင် အရင်လုပ်ပါမယ်
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
            h, w = processed.shape
            
            num_rows = st.sidebar.number_input("Rows", value=25)
            num_cols = st.sidebar.selectbox("Cols", [2,4,6,8], index=3)
            col_w = w / num_cols
            grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]

            # OCR ဖတ်ခြင်း
            results = reader.readtext(processed, detail=1)

            for (bbox, text, prob) in results:
                # နေရာတွက်ချက်ခြင်း
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                c_idx, r_idx = int(cx / col_w), int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    # OCR က သိပ်မသေချာရင် Template နဲ့ ထပ်စစ်မယ်
                    if prob < 0.5: 
                        x_min, y_min = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
                        x_max, y_max = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
                        roi = processed[y_min:y_max, x_min:x_max]
                        matched = match_digit_from_image(roi)
                        if matched: text = matched

                    # စာသားသန့်စင်ခြင်း
                    txt = clean_ocr_text(text)
                    
                    if c_idx % 2 == 0: # Number Column
                        nums = re.findall(r'\d+', txt)
                        grid[r_idx][c_idx] = nums[0].zfill(3) if nums else txt
                    else: # Amount Column
                        nums = re.findall(r'\d+', txt)
                        grid[r_idx][c_idx] = nums[0] if nums else ""

            # Ditto Logic (အပေါ်ကွက်အတိုင်း အောက်ကွက်ဖြည့်ခြင်း)
            for c in range(num_cols):
                last_val = ""
                for r in range(num_rows):
                    if grid[r][c] == "" and last_val != "":
                        grid[r][c] = last_val
                    elif grid[r][c] != "":
                        last_val = grid[r][c]

            st.session_state['data'] = grid
            st.success("✅ အကုန်ဖတ်ပြီးပါပြီ။")

# ---------------- DISPLAY ----------------
if 'data' in st.session_state:
    edited = st.data_editor(st.session_state['data'], use_container_width=True)
    # Upload button logic ဆက်ရေးရန်...
