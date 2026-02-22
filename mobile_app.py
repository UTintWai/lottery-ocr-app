import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro 2026 Stable", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- GRID PROCESSING ---
def process_grid(img, n_rows, n_cols):
    h, w = img.shape[:2]
    results = reader.readtext(img, detail=1)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        c_idx = int(cx / (w / n_cols))
        r_idx = int(cy / (h / n_rows))
        
        if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
            val = text.strip()
            # Ditto Logic (။) အမှတ်အသားများ
            if any(m in val for m in ['"', '။', '=', '||', '..', '`', '4', 'u', 'U']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                clean_num = re.sub(r'[^0-9\*xX]', '', val)
                if clean_num.isdigit() and len(clean_num) < 3:
                    clean_num = clean_num.zfill(3)
                grid[r_idx][c_idx] = clean_num

    for c in range(n_cols):
        for r in range(1, n_rows):
            if grid[r][c] == "DITTO":
                grid[r][c] = grid[r-1][c]
    return grid

# --- UI ---
st.title("🎯 Lottery Pro (Google Sheets Fix)")

with st.sidebar:
    st.header("⚙️ Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    target_sheet_name = st.radio("ပို့မည့် Sheet", ["Sheet1", "Sheet2", "Sheet3"])

uploaded_file = st.file_uploader("လက်ရေး Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(img, width=400, caption="မူရင်းပုံ")

    if st.button("🔍 Scan Table"):
        with st.spinner("၈ တိုင်လုံးကို အကွက်ကျကျ ဖတ်နေပါသည်..."):
            final_grid = process_grid(img, n_rows, a_cols)
            st.session_state['processed_data'] = final_grid

if 'processed_data' in st.session_state:
    st.subheader("📝 Scan ရလဒ် (ပြင်ဆင်ပြီးမှ ပို့ပါ)")
    edited_data = st.data_editor(st.session_state['processed_data'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheets"):
        try:
            # ၁။ Secrets စစ်ဆေးခြင်း
            if "GCP_SERVICE_ACCOUNT_FILE" not in st.secrets:
                st.error("Error: 'GCP_SERVICE_ACCOUNT_FILE' ကို Secrets ထဲမှာ မတွေ့ပါ။")
                st.stop()
            
            # ၂။ Connection အဆင့်
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
            client = gspread.authorize(creds)
            
            # ၃။ Sheet ဖွင့်ခြင်း (နာမည်ကို သေချာစစ်ပါ)
            ss = client.open("LotteryData")
            sh = ss.worksheet(target_sheet_name)
            
            # ၄။ ဒေတာပြင်ဆင်ခြင်း (ရှေ့က 0 မပျောက်ရန် ' ခံခြင်း)
            formatted_data = [[f"'{cell}" if str(cell).strip() != "" else "" for cell in row] for row in edited_data]
            clean_rows = [r for r in formatted_data if any(c != "" for c in r)]
            
            if clean_rows:
                sh.append_rows(clean_rows, value_input_option='USER_ENTERED')
                st.success(f"✅ {len(clean_rows)} တန်းကို {target_sheet_name} ထဲသို့ အောင်မြင်စွာ ပို့ပြီးပါပြီ!")
            else:
                st.warning("ပို့ရန် ဒေတာ မရှိပါ။")

        except gspread.exceptions.SpreadsheetNotFound:
            st.error("Error: 'LotteryData' ဆိုတဲ့ Google Sheet ကို မတွေ့ပါ။ Sheet နာမည် မှန်အောင် ပြင်ပေးပါဗျ။")
        except gspread.exceptions.WorksheetNotFound:
            st.error(f"Error: {target_sheet_name} ဆိုတဲ့ Tab ကို မတွေ့ပါ။")
        except Exception as e:
            st.error(f"Error: {str(e)}")
