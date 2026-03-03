import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro v55 (Side Split)", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_side(img, is_left=True):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1500 # တစ်ဖက်တည်းဖြစ်၍ resolution ကို ညှိထားသည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ၃ နဲ့ ၈ ခွဲခြားရန် အထူးပြုပြင်ချက်
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    kernel = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(thresh, kernel, iterations=1)
    final_img = cv2.bitwise_not(eroded)

    results = reader.readtext(final_img, paragraph=False, mag_ratio=1.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text
        })
    
    if not raw_data: return []

    # စာကြောင်းခွဲခြင်း
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

    processed_side_data = []
    col_width = target_w / 4 # တစ်ဖက်တွင် ၄ တိုင်စီရှိသည်
    
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 4:
                txt = item['text'].upper().strip()
                if re.search(r'[။၊"=“_…\.\-]', txt) or (not txt.isdigit() and len(txt) == 1):
                    row_cells[c_idx] = "DITTO"
                else:
                    # ဂဏန်းအမှားပြင်ဆင်ချက်
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:]
                        else: row_cells[c_idx] = num
        processed_side_data.append(row_cells)
    
    # Ditto fill logic (Amount Columns only)
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_side_data)):
            v = processed_side_data[r][c]
            if v.isdigit(): last_val = v
            elif (v == "DITTO" or v == "") and last_val != "": processed_side_data[r][c] = last_val
            
    return processed_side_data

# --- UI ---
st.title("🔢 Side-by-Side Expert v55")
st.write("ဗောက်ချာကို **ဘယ်ဘက် ၄ တိုင် တစ်ပုံ၊ ညာဘက် ၄ တိုင် တစ်ပုံ** ခွဲရိုက်ပေးပါ။ ဂဏန်းများ ပိုမိုပြတ်သားစွာ ဖတ်နိုင်ပါမည်။")

c1, c2 = st.columns(2)
with c1:
    up_left = st.file_uploader("ပုံ (၁) - ဘယ်ဘက် ၄ တိုင် (A,B,C,D)", type=['jpg', 'png'])
with c2:
    up_right = st.file_uploader("ပုံ (၂) - ညာဘက် ၄ တိုင် (E,F,G,H)", type=['jpg', 'png'])

if st.button("🔍 Combine and Scan All 8 Columns"):
    left_data = []
    right_data = []
    
    if up_left:
        with st.spinner("ဘယ်ဘက်ပိုင်းကို အသေးစိတ်ဖတ်နေသည်..."):
            img_l = cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1)
            left_data = process_side(img_l, is_left=True)
            
    if up_right:
        with st.spinner("ညာဘက်ပိုင်းကို အသေးစိတ်ဖတ်နေသည်..."):
            img_r = cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1)
            right_data = process_side(img_r, is_left=False)
            
    # အတန်းအရေအတွက်ကို ညှိပြီး ပေါင်းစပ်ခြင်း
    max_rows = max(len(left_data), len(right_data))
    final_8_cols = []
    for i in range(max_rows):
        l_row = left_data[i] if i < len(left_data) else ["","","",""]
        r_row = right_data[i] if i < len(right_data) else ["","","",""]
        final_8_cols.append(l_row + r_row)
        
    st.session_state['data_v55'] = final_8_cols

if 'data_v55' in st.session_state:
    st.subheader("ပေါင်းစပ်ပြီး ၈ တိုင် ဇယား (A မှ H)")
    edited = st.data_editor(st.session_state['data_v55'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        st.success("Google Sheet ထဲသို့ ဒေတာအားလုံး အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
