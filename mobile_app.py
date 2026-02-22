import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro 2026 Ultimate", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- IMAGE PROCESSING FOR 8 COLUMNS ---
def process_grid_fixed(img, n_rows, n_cols):
    # ပုံကို Gray ပြောင်းပြီး အစွန်းတွေကို ချဲ့လိုက်ခြင်း (Padding)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ဘေးဘောင်တွေကို အဖြူသား ၂၀ pixel စီ ချဲ့လိုက်လို့ ဘေးဆုံးက စာတွေ မလွတ်တော့ပါဘူး
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    
    h, w = gray.shape
    results = reader.readtext(gray, detail=1)
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    
    # ၈ တိုင်အတွက် Column width ကို တိကျစွာ ခွဲဝေခြင်း
    col_width = w / n_cols
    row_height = h / n_rows

    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        c_idx = int(cx / col_width)
        r_idx = int(cy / row_height)
        
        if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
            val = text.strip()
            # Ditto (။) စစ်ဆေးခြင်း
            if any(m in val for m in ['"', '။', '=', '||', '..', '`', '4', 'u', 'U']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                # ဂဏန်း ၃ လုံး (0) ဖြည့်ခြင်း
                clean_num = re.sub(r'[^0-9\*xX]', '', val)
                if clean_num.isdigit() and len(clean_num) < 3:
                    clean_num = clean_num.zfill(3)
                grid[r_idx][c_idx] = clean_num

    # Ditto Fill Logic
    for c in range(n_cols):
        for r in range(1, n_rows):
            if grid[r][c] == "DITTO":
                grid[r][c] = grid[r-1][c]
    return grid

# --- UI ---
st.title("🎯 Lottery Pro (8-Column Fix)")

with st.sidebar:
    st.header("⚙️ Settings")
    # ၈ တိုင်ကို ပုံသေရွေးထားပေးပါမယ်
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(img, width=500, caption="မူရင်းပုံ")

    if st.button("🔍 Scan 8 Columns"):
        with st.spinner("၈ တိုင်စလုံးကို အနားသတ်အပြည့် ဖတ်နေပါသည်..."):
            final_grid = process_grid_fixed(img, n_rows, a_cols)
            st.session_state['processed_data'] = final_grid

if 'processed_data' in st.session_state:
    st.subheader("📝 Scan ရလဒ် (၈ တိုင်)")
    edited_data = st.data_editor(st.session_state['processed_data'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheets"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh = ss.worksheet("Sheet1")
            
            # Formatting for zero
            formatted_data = [[f"'{cell}" if str(cell).strip() != "" else "" for cell in row] for row in edited_data]
            clean_rows = [r for r in formatted_data if any(c != "" for c in r)]
            
            if clean_rows:
                sh.append_rows(clean_rows, value_input_option='USER_ENTERED')
                st.success("✅ ၈ တိုင်စလုံး Sheet ထဲသို့ ပို့ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
