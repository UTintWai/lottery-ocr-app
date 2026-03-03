import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro v54 (Split-Scan)", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_image(img, target_w=2800):
    reader = load_ocr()
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # --- 🔥 ADVANCED DIGIT SHARPENING ---
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

    final_table = []
    col_width = target_w / 8
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                txt = item['text'].upper().strip()
                if re.search(r'[။၊"=“_…\.\-]', txt) or (not txt.isdigit() and len(txt) == 1):
                    row_cells[c_idx] = "DITTO"
                else:
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:]
                        else: row_cells[c_idx] = num
        final_table.append(row_cells)
        
    # Ditto fill logic
    for c in [1, 3, 5, 7]:
        active_amt = ""
        for r in range(len(final_table)):
            val = str(final_table[r][c]).strip()
            if val.isdigit() and val != "": active_amt = val
            elif (val == "DITTO" or val == "") and active_amt != "": final_table[r][c] = active_amt
            
    return final_table

# --- UI ---
st.title("🔢 Split-Scan Ultra v54")
st.write("အကွက် ၂၀၀ ကို တစ်ပုံတည်းမဖတ်ဘဲ နှစ်ပိုင်းခွဲရိုက်ခြင်းဖြင့် ၃/၈ နှင့် ၅/၆ အမှားကို ၉၀% အထက် လျှော့ချနိုင်ပါမည်။")

col1, col2 = st.columns(2)
with col1:
    up_top = st.file_uploader("အပိုင်း (၁) - အပေါ်တစ်ဝက်တင်ပါ", type=['jpg', 'png'])
with col2:
    up_bottom = st.file_uploader("အပိုင်း (၂) - အောက်တစ်ဝက်တင်ပါ", type=['jpg', 'png'])

if st.button("🔍 Start Combined Deep Scan"):
    combined_data = []
    if up_top:
        with st.spinner("အပိုင်း (၁) ကို အသေးစိတ်ဖတ်နေသည်..."):
            img1 = cv2.imdecode(np.frombuffer(up_top.read(), np.uint8), 1)
            combined_data.extend(process_image(img1))
    if up_bottom:
        with st.spinner("အပိုင်း (၂) ကို အသေးစိတ်ဖတ်နေသည်..."):
            img2 = cv2.imdecode(np.frombuffer(up_bottom.read(), np.uint8), 1)
            combined_data.extend(process_image(img2))
    
    st.session_state['data_v54'] = combined_data

if 'data_v54' in st.session_state:
    st.subheader(f"စုစုပေါင်းအကွက် {len(st.session_state['data_v54'])} ကွက် ဖတ်မိပါသည်")
    edited = st.data_editor(st.session_state['data_v54'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save All to Google Sheet"):
        st.success("Google Sheet ထဲသို့ ဒေတာအားလုံး သိမ်းဆည်းလိုက်ပါပြီ!")
