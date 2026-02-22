import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026 - Zero & Ditto Fix", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def advanced_processing(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # လက်ရေးပိုပြတ်သားစေရန် adaptiveThreshold ကိုသုံးခြင်း
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    return thresh

def format_three_digits(txt):
    """ဂဏန်းသက်သက်ဆိုလျှင် ၃ လုံးပြည့်အောင် ရှေ့က 0 ဖြည့်ပေးရန်"""
    if txt.isdigit() and len(txt) < 3:
        return txt.zfill(3)
    return txt

def is_ditto(txt):
    """Ditto (။ ။) သို့မဟုတ် အလားတူ အမှတ်အသားများကို စစ်ဆေးရန်"""
    ditto_marks = ['"', '။', '=', '||', '..', '`', '“']
    return any(mark in txt for mark in ditto_marks)

# --- UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    n_rows = st.number_input("Rows", min_value=1, value=25)
    a_cols = st.selectbox("Columns", [2, 4, 6, 8], index=3)

uploaded_file = st.file_uploader("လက်ရေး Voucher ပုံတင်ပါ", type=["jpg","jpeg","png"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    if st.button("🔍 Scan & Auto-Fill"):
        processed_img = advanced_processing(img)
        h, w = processed_img.shape
        grid_data = [["" for _ in range(a_cols)] for _ in range(n_rows)]
        
        results = reader.readtext(processed_img, detail=1)
        
        col_edges = np.linspace(0, w, a_cols + 1)
        row_edges = np.linspace(0, h, n_rows + 1)

        for (bbox, text, prob) in results:
            cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
            c_idx, r_idx = np.searchsorted(col_edges, cx) - 1, np.searchsorted(row_edges, cy) - 1
            
            if 0 <= r_idx < n_rows and 0 <= c_idx < a_cols:
                raw_txt = text.strip()
                
                if is_ditto(raw_txt):
                    grid_data[r_idx][c_idx] = "DITTO"
                else:
                    # ဂဏန်းနှင့် * ကိုပဲ ယူမည်
                    clean_val = re.sub(r'[^0-9\*xX]', '', raw_txt).replace('x', '*').replace('X', '*')
                    grid_data[r_idx][c_idx] = format_three_digits(clean_val)

        # --- Ditto Fill Logic (အပေါ်ကတန်ဖိုးကို ပြန်ဖြည့်ခြင်း) ---
        for c in range(a_cols):
            for r in range(1, n_rows):
                if grid_data[r][c] == "DITTO":
                    grid_data[r][c] = grid_data[r-1][c]

        st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 Scan ရလဒ် (0 ဖြည့်ပြီး၊ Ditto ပေါင်းပြီး)")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheets"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            sh = ss.worksheet("Sheet1")
            
            # ဒေတာတွေကို ပို့တဲ့အခါ ရှေ့က 0 မပျောက်အောင် ' (Apostrophe) လေး ခံပေးခြင်း
            formatted_to_send = []
            for row in edited_data:
                formatted_to_send.append([f"'{cell}" if cell != "" else "" for cell in row])
            
            sh.append_rows(formatted_to_send, value_input_option='USER_ENTERED')
            st.success("✅ Google Sheets ထဲသို့ (0) များမပျောက်ဘဲ ပို့ဆောင်ပြီးပါပြီ!")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
