import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG & OCR ENGINE ---
st.set_page_config(page_title="AI Lottery Scanner Pro", layout="wide")

@st.cache_resource
def load_ocr():
    # Cloud မှာ model download ဆွဲရလွယ်အောင် model_storage_directory ထည့်ထားပါတယ်
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models')

reader = load_ocr()

# --- IMAGE PROCESSING ---
def pre_process_for_lottery(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # စာလုံးကြည်အောင် ၁.၅ ဆပဲ ချဲ့ပါမယ် (RAM ချွေတာရန်)
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
    # GaussianBlur က fastNlMeans ထက် ပိုမြန်ပြီး RAM အစားသက်သာပါတယ်
    dist = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh

def get_lottery_data(img, rows, cols):
    processed_img = pre_process_for_lottery(img)
    h, w = processed_img.shape
    results = reader.readtext(processed_img, detail=1, paragraph=False)
    
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        cx, cy = (tl[0] + br[0]) / 2, (tl[1] + br[1]) / 2
        
        c_idx = int(cx / (w / cols))
        r_idx = int(cy / (h / rows))
        
        if 0 <= r_idx < rows and 0 <= c_idx < cols:
            val = text.strip().upper()
            # DITTO သတ်မှတ်ချက်များ
            if any(char in val for char in ['"', '။', '=', 'U', 'V', '`', '4']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                clean_num = re.sub(r'[^0-9]', '', val)
                if clean_num:
                    grid[r_idx][c_idx] = clean_num.zfill(3)

    # DITTO Fill Down Logic
    for c in range(cols):
        for r in range(1, rows):
            if grid[r][c] == "DITTO" and grid[r-1][c] != "":
                grid[r][c] = grid[r-1][c]
    return grid

# --- GOOGLE SHEETS FUNCTION ---
def save_to_sheets(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        # Streamlit Secrets ကနေ Key ကို ယူမှာဖြစ်ပါတယ်
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # သင့် Google Sheet အမည်ကို 'LotteryData' လို့ ပေးထားရပါမယ်
        sheet = client.open("LotteryData").sheet1
        
        # ဒေတာအလွတ်တွေကို ဖယ်ပြီး ပို့မယ်
        clean_rows = [r for r in data if any(c != "" for c in r)]
        if clean_rows:
            sheet.append_rows(clean_rows)
            return True
    except Exception as e:
        st.error(f"Error connecting to Google Sheets: {e}")
        return False

# --- UI ---
st.title("🔢 AI Lottery Scanner (6/8 Columns)")

with st.sidebar:
    st.header("Settings")
    col_count = st.selectbox("တိုင်အရေအတွက် ရွေးပါ", [6, 8], index=1)
    row_count = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    st.info("Google Sheet အမည်ကို 'LotteryData' ဟု ပေးထားရန် လိုအပ်ပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    raw_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="တင်ထားသောပုံ", width=350)
    
    if st.button("🚀 စကင်ဖတ်မယ်"):
        with st.spinner("စာလုံးများကို ဖော်ထုတ်နေပါသည်..."):
            final_data = get_lottery_data(img, row_count, col_count)
            st.session_state['scan_result'] = final_data

if 'scan_result' in st.session_state:
    st.subheader("စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_data = st.data_editor(st.session_state['scan_result'], use_container_width=True)
    
    if st.button("💾 Google Sheet ထဲသိမ်းမယ်"):
        with st.spinner("Sheets ထဲသို့ ပို့နေပါသည်..."):
            if save_to_sheets(edited_data):
                st.success("✅ Google Sheet ထဲသို့ ဒေတာများ ပို့ဆောင်ပြီးပါပြီ!")
