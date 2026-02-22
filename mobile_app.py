import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026 Ultimate", layout="wide")

# --- OCR Load ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- Image Cleaning & Pre-processing ---
def advanced_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def clean_ocr_text(txt):
    txt = txt.upper().strip()
    repls = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'7','T':'7','Q':'0','D':'0'}
    for k,v in repls.items():
        txt = txt.replace(k,v)
    return txt

# --- Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    n_rows = st.number_input("Rows (အတန်း)", min_value=1, value=25)
    a_cols = st.selectbox("Columns (အတိုင်)", [2, 4, 6, 8], index=3)
    target_sheet = st.selectbox("ပို့မည့် Sheet ရွေးပါ", ["Sheet1", "Sheet2", "Sheet3"])

# --- Main UI ---
uploaded_file = st.file_uploader("လက်ရေး Voucher ပုံတင်ပါ", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="မူရင်းပုံ", use_container_width=True)
    
    if st.button("🔍 Scan စတင်မည်"):
        processed_img = advanced_processing(img)
        with col2:
            st.image(processed_img, caption="OCR အတွက် အချောကိုင်ထားသောပုံ", use_container_width=True)
        
        h, w = processed_img.shape
        grid_data = [["" for _ in range(a_cols)] for _ in range(n_rows)]
        
        results = reader.readtext(processed_img, detail=1)
        
        col_edges = np.linspace(0, w, a_cols + 1)
        row_edges = np.linspace(0, h, n_rows + 1)

        for (bbox, text, prob) in results:
            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox])
            
            c_idx = np.searchsorted(col_edges, cx) - 1
            r_idx = np.searchsorted(row_edges, cy) - 1
            
            if 0 <= r_idx < n_rows and 0 <= c_idx < a_cols:
                txt = clean_ocr_text(text)
                match = re.search(r'[\d\*\.xX]+', txt)
                if match:
                    clean_val = match.group().replace('X', '*').replace('x', '*')
                    grid_data[r_idx][c_idx] = clean_val

        st.session_state['data_final'] = grid_data

# --- Data Display & Send to Sheets ---
if 'data_final' in st.session_state:
    st.subheader("📝 ဇယားရလဒ် (ပြင်ဆင်ပြီးပါက ပို့ရန်နှိပ်ပါ)")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheets"):
        try:
            # Google Sheets API Connection
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh = ss.worksheet(target_sheet)
            
            # ဇယားကွက်ထဲမှ အကွက်လွတ်မဟုတ်သော ဒေတာများကိုသာ ပို့မည်
            sh.append_rows(edited_data)
            st.success(f"✅ {target_sheet} ထဲသို့ ဒေတာများကို ဇယားအတိုင်း အောင်မြင်စွာ ပို့ပြီးပါပြီ!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
