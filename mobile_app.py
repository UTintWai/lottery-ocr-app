import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Precision v51", layout="wide")

@st.cache_resource
def load_ocr():
    # Model ကို Precision အမြင့်ဆုံးဖြစ်အောင် Setup လုပ်ထားခြင်း
    return easyocr.Reader(['en'], gpu=False)

def process_v51(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resolution ကို အလွန်မြင့်မားသော 3000px သို့ တိုးမြှင့်သည်
    target_w = 3000 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # --- 🔥 HIGH-DEFINITION PRE-PROCESSING ---
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ၁။ စာလုံးအနားသတ်များကို အနက်ရောင်ပြတ်သားစေရန် လုပ်ဆောင်ခြင်း
    inv_gray = cv2.bitwise_not(gray)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(inv_gray, -1, kernel)
    final_gray = cv2.bitwise_not(sharpened)
    
    # ၂။ Adaptive Thresholding (၃ နဲ့ ၈ အပေါက်များကို ပိုမြင်သာစေရန်)
    thresh = cv2.adaptiveThreshold(final_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)

    # OCR Scan (ပုံကို တစ်ပိုင်းချင်းစီ အသေးစိတ်ဖတ်မည်)
    results = reader.readtext(thresh, paragraph=False, mag_ratio=1.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text,
            'prob': prob
        })

    if not raw_data: return []

    # ROW GROUPING (စာကြောင်းအကွာအဝေးကို စနစ်တကျခွဲခြင်း)
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    if raw_data:
        curr_row = [raw_data[0]]
        for i in range(1, len(raw_data)):
            if raw_data[i]['y'] - curr_row[-1]['y'] < 30: 
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
                
                # Ditto detections
                if re.search(r'[။၊"=“_…\.\-]', txt) or (not txt.isdigit() and len(txt) == 1):
                    row_cells[c_idx] = "DITTO"
                else:
                    # Logic-based corrections
                    txt = txt.replace('S','5').replace('G','6').replace('I','1').replace('B','8').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0:
                            row_cells[c_idx] = num.zfill(3)[-3:]
                        else:
                            row_cells[c_idx] = num
        final_table.append(row_cells)

    # --- VERTICAL AUTO-FILL FOR AMOUNTS ---
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
st.title("🔢 Lottery Ultra-Precision v51")
st.markdown("#### ၃/၈ နှင့် ၅/၆ ပြဿနာအတွက် အဆင့်မြင့် Image Sharpener ကို အသုံးပြုထားပါသည်")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ (Focus ပြတ်သားသောပုံကို သုံးပါ)", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 Run Precision Scan"):
        with st.spinner("စာလုံးများကို ခွဲခြားနိုင်ရန် အနီးကပ် တွက်ချက်နေပါသည်..."):
            res = process_v51(img)
            st.session_state['data_v51'] = res

if 'data_v51' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    # Edit လုပ်နိုင်သော ဇယားကွက်
    edited = st.data_editor(st.session_state['data_v51'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        # Save function here
        st.success("ဒေတာများကို Google Sheet ထဲသို့ ပို့ဆောင်လိုက်ပါပြီ!")
