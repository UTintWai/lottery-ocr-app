import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v2", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM ချွေတာရန် recognition model ကို သီးသန့် ညွှန်ကြားထားပါသည်
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models', recog_network='english_g2')

reader = load_ocr()

def process_image_smart(img, rows, cols):
    # 1. Image Resize (RAM Crash မဖြစ်စေရန် အရွယ်အစားလျှော့ခြင်း)
    h, w = img.shape[:2]
    max_dim = 1500
    if h > max_dim or w > max_dim:
        scale = max_dim / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OCR ဖတ်ခြင်း
    results = reader.readtext(gray)
    
    new_h, new_w = gray.shape
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    # Column နဲ့ Row width ကို တွက်ချက်ခြင်း
    col_w = new_w / cols
    row_h = new_h / rows

    for (bbox, text, prob) in results:
        # Bounding box ဗဟိုကို ယူခြင်း
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        c_idx = int(cx // col_w)
        r_idx = int(cy // row_h)
        
        if 0 <= r_idx < rows and 0 <= c_idx < cols:
            val = text.strip().upper()
            # DITTO စစ်ဆေးခြင်း
            if any(m in val for m in ['"', '။', '=', 'U', 'V', '`', '4']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                # ဂဏန်း ၃ လုံး သီးသန့် ယူခြင်း
                num = re.sub(r'[^0-9]', '', val)
                if num:
                    grid[r_idx][c_idx] = num.zfill(3)

    # DITTO Fill Down
    for c in range(cols):
        for r in range(1, rows):
            if grid[r][c] == "DITTO" and grid[r-1][c] != "":
                grid[r][c] = grid[r-1][c]
                
    return grid

def save_to_sheets(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        # Google Sheet အမည်မှာ 'LotteryData' ဖြစ်ရပါမည်
        sheet = client.open("LotteryData").sheet1
        
        # အလွတ်တန်းများ ဖယ်ထုတ်ခြင်း
        clean_rows = [r for r in data if any(c != "" for c in r)]
        # Google Sheet ထဲတွင် 0 အရှေ့က မပျောက်စေရန် ' ထည့်ပေးခြင်း
        formatted_data = [[f"'{c}" if c != "" else "" for c in row] for row in clean_rows]
        
        if formatted_data:
            sheet.append_rows(formatted_data)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Lottery Scanner (RAM Optimized)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

up_file = st.file_uploader("Voucher ပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    # ပုံကို RAM သက်သက်သာသာ ဖတ်ခြင်း
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400, caption="မူရင်းပုံ")
    
    if st.button("🔍 Scan လုပ်မယ်"):
        with st.spinner("ဖတ်နေပါသည်..."):
            final_grid = process_image_smart(img, n_rows, a_cols)
            st.session_state['scan_data'] = final_grid

if 'scan_data' in st.session_state:
    st.subheader("စစ်ဆေးရန်")
    edited_df = st.data_editor(st.session_state['scan_data'], use_container_width=True)
    
    if st.button("💾 Google Sheet သို့ ပို့မယ်"):
        if save_to_sheets(edited_df):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!")
