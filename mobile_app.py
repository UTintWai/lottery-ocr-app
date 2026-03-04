import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
from io import BytesIO

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v66", layout="wide")
SHEET_NAME = "LotteryData" 

def save_to_gsheet(data_to_save):
    try:
        if not data_to_save: return False
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets ထဲမှာ Google Key မရှိသေးပါဗျ။")
            return False
            
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).get_worksheet(0)
        
        # Format strings to preserve leading zeros
        formatted_rows = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data_to_save]
        
        if formatted_rows:
            sheet.append_rows(formatted_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_v66(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Pre-processing for poor quality photos
    dst = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(dst)
    
    results = reader.readtext(enhanced, paragraph=False, mag_ratio=2.0)
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 'text': t} for (b, t, p) in results]
    elements.sort(key=lambda k: k['y'])

    rows = []
    if elements:
        curr_row = [elements[0]]
        for i in range(1, len(elements)):
            if elements[i]['y'] - curr_row[-1]['y'] < 35:
                curr_row.append(elements[i])
            else:
                rows.append(curr_row)
                curr_row = [elements[i]]
        rows.append(curr_row)

    processed_data = []
    for r_items in rows:
        r_items.sort(key=lambda k: k['x'])
        row_cells = ["" for _ in range(4)]
        for i, item in enumerate(r_items[:4]):
            txt = item['text'].upper().strip()
            if re.search(r'[။၊"=“_…\.\-\']', txt):
                row_cells[i] = "DITTO"
            else:
                txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if i % 2 == 0: row_cells[i] = num.zfill(3)[-3:]
                    else: row_cells[i] = num
        processed_data.append(row_cells)

    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit() and v != "": last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v66 (Sync & Backup)")
st.info("ပုံအရည်အသွေးကို အလိုအလျောက်မြှင့်တင်ပေးပြီး Backup CSV ထုတ်ယူနိုင်ပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ပုံ (၁) - ဘယ်ဘက်", type=['jpg', 'png', 'jpeg'])
with c2: up_right = st.file_uploader("ပုံ (၂) - ညာဘက်", type=['jpg', 'png', 'jpeg'])

if st.button("🔍 Start Scanning"):
    l_data, r_data = [], []
    if up_left: l_data = process_v66(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
    if up_right: r_data = process_v66(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
    max_r = max(len(l_data), len(r_data))
    final = [(l_data[i] if i < len(l_data) else ["","","",""]) + (r_data[i] if i < len(r_data) else ["","","",""]) for i in range(max_r)]
    st.session_state['v66_data'] = final

if 'v66_data' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    edited = st.data_editor(st.session_state['v66_data'], use_container_width=True, num_rows="dynamic")
    
    col_save, col_csv = st.columns(2)
    
    with col_save:
        if st.button("💾 Save to Google Sheet"):
            if save_to_gsheet(edited):
                st.success("✅ Google Sheet ထဲ ပို့ပြီးပါပြီ!")
                st.balloons()
    
    with col_csv:
        # CSV Backup logic
        df = pd.DataFrame(edited, columns=['A','B','C','D','E','F','G','H'])
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download as CSV (Backup)",
            data=csv,
            file_name='lottery_backup.csv',
            mime='text/csv',
        )
