import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Lottery Pro v58", layout="wide")
SHEET_NAME = "LotteryData"  # သင့် Sheet နာမည်နဲ့ ကိုက်အောင်ပြင်ပါ

# --- 2. GOOGLE SHEETS CONNECTION ---
def save_to_gsheet(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Streamlit Secrets ထဲမှာ credentials ရှိမရှိစစ်သည်
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets ထဲမှာ Credentials မရှိသေးပါဗျ။ Settings > Secrets မှာ ထည့်ပေးပါ။")
            return False
            
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Sheet ကိုဖွင့်သည်
        sheet = client.open(SHEET_NAME).sheet1
        
        # ဒေတာများကို အောက်ဆုံးကနေ ဆက်ထည့်သည်
        sheet.append_rows(data, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Sheet Error: {e}")
        return False

# --- 3. OCR PROCESSING ENGINE ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_side(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1500 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ၃ နဲ့ ၈ ခွဲခြားရန် Image Processing
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

    processed_side_data = []
    col_width = target_w / 4 
    
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 4:
                txt = item['text'].upper().strip()
                if re.search(r'[။၊"=“_…\.\-]', txt) or (not txt.isdigit() and len(txt) == 1):
                    row_cells[c_idx] = "DITTO"
                else:
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:]
                        else: row_cells[c_idx] = num
        processed_side_data.append(row_cells)
    
    # Amount Columns အတွက် Ditto fill လုပ်ခြင်း
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_side_data)):
            v = str(processed_side_data[r][c])
            if v.isdigit() and v != "": last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_side_data[r][c] = last_val
            
    return processed_side_data

# --- 4. USER INTERFACE (UI) ---
st.title("🔢 Side-by-Side Expert v58")
st.write("ဘယ် ၄ တိုင် တစ်ပုံ၊ ညာ ၄ တိုင် တစ်ပုံ ခွဲတင်ပေးပါဗျ။")

col1, col2 = st.columns(2)
with col1:
    up_left = st.file_uploader("ပုံ (၁) - ဘယ် ၄ တိုင်", type=['jpg', 'png'])
with col2:
    up_right = st.file_uploader("ပုံ (၂) - ညာ ၄ တိုင်", type=['jpg', 'png'])

if st.button("🔍 Combine and Scan"):
    left_data, right_data = [], []
    
    if up_left:
        with st.spinner("ဘယ်ဘက်ပိုင်းကို ဖတ်နေသည်..."):
            img_l = cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1)
            left_data = process_side(img_l)
            
    if up_right:
        with st.spinner("ညာဘက်ပိုင်းကို ဖတ်နေသည်..."):
            img_r = cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1)
            right_data = process_side(img_r)
            
    max_rows = max(len(left_data), len(right_data))
    final_8_cols = []
    for i in range(max_rows):
        l_row = left_data[i] if i < len(left_data) else ["","","",""]
        r_row = right_data[i] if i < len(right_data) else ["","","",""]
        final_8_cols.append(l_row + r_row)
        
    st.session_state['combined_data'] = final_8_cols

# --- 5. DATA TABLE & SAVE ---
if 'combined_data' in st.session_state:
    st.subheader("ပေါင်းစပ်ပြီး ၈ တိုင် ဇယား")
    edited_data = st.data_editor(st.session_state['combined_data'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        with st.spinner("Sheet ထဲသို့ ပို့နေပါသည်..."):
            if save_to_gsheet(edited_data):
                st.success("✅ Google Sheet ထဲသို့ အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
                st.balloons()
