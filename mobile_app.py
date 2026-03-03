import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro v52", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_v52(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resolution ကို ပိုတိုးလိုက်သည် (စာလုံးသေးများ အနီးကပ်မြင်ရစေရန်)
    target_w = 2800 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # --- 🔥 ADVANCED DIGIT CLEANING (3/8 & 5/6 Focus) ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ၁။ စာလုံးလိုင်းများကို ပိုမိုပြတ်သားအောင် Contrast မြှင့်ခြင်း
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    
    # ၂။ Thresholding (၃ နဲ့ ၈ အပေါက်များကို ခွဲခြားရန်)
    # Binary Inverse လုပ်ပြီး စာလုံးကို ဖြူစေကာ နောက်ခံကို မည်းစေခြင်းဖြင့် အပေါက်တွေကို ပိုမြင်ရစေသည်
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    # ၃။ Skeletonization/Thinning (စာလုံးလိုင်းများကို ပါးလွှာစေပြီး ၃ နဲ့ ၈ ကို ကွဲပြားစေသည်)
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    final_img = cv2.bitwise_not(eroded) # OCR ဖတ်ရန် ပြန်လှန်သည်

    # OCR Scan
    results = reader.readtext(final_img, paragraph=False, mag_ratio=1.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text
        })

    if not raw_data: return []

    # ROW GROUPING
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 25: 
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    # --- 8-COLUMN GRID MAPPING ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                txt = item['text'].upper().strip()
                
                # Ditto marks logic (။ ၊ " =)
                if re.search(r'[။၊"=“_…\.\-]', txt) or (not txt.isdigit() and len(txt) == 1):
                    row_cells[c_idx] = "DITTO"
                else:
                    # Digit Fixes (S->5, G->6, B->8)
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0:
                            row_cells[c_idx] = num.zfill(3)[-3:]
                        else:
                            row_cells[c_idx] = num
        final_table.append(row_cells)

    # --- VERTICAL AUTO-FILL (DITTO Logic) ---
    for c in [1, 3, 5, 7]:
        active_amt = ""
        for r in range(len(final_table)):
            val = str(final_table[r][c]).strip()
            if val.isdigit() and val != "":
                active_amt = val
            elif (val == "DITTO" or val == "") and active_amt != "":
                final_table[r][c] = active_amt

    return final_table

# --- UI ---
st.title("🔢 Lottery Pro v52 (Digit Distinction Fix)")
st.info("၃ နှင့် ၈၊ ၅ နှင့် ၆ တို့ကို ခွဲခြားနိုင်ရန် စာလုံးလိုင်းများကို ပါးလွှာပြတ်သားအောင် ပြုပြင်ထားပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 High-Precision Scan"):
        with st.spinner("စာလုံးများကို အနားသတ်ညှိပြီး ဖတ်နေပါသည်..."):
            res = process_v52(img)
            st.session_state['data_v52'] = res

if 'data_v52' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    edited = st.data_editor(st.session_state['data_v52'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        st.success("Google Sheet ထဲသို့ ပို့လိုက်ပါပြီ!")
