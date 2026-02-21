import streamlit as st
import numpy as np
import easyocr
import cv2
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro 2026 Stable", layout="wide")

@st.cache_resource
def load_ocr():
    # CPU သုံးသူများအတွက် အမြန်ဆုံး mode ဖြစ်အောင် ချိန်ညှိထားသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- SCAN FUNCTION (၈ တိုင် အမြန်ဖတ်ရန်) ---
def scan_voucher_final(img, active_cols, num_rows):
    # ၁။ ပုံကို OCR ဖတ်ရ ပိုလွယ်အောင် Grayscale ပြောင်းပြီး Contrast မြှင့်တင်ခြင်း
    img_resized = cv2.resize(img, (0,0), fx=0.4, fy=0.4) 
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ပုံ၏ အရွယ်အစားကို တိတိကျကျ ယူခြင်း
    h, w = gray.shape

    # ၂။ OCR ဖတ်ခြင်း (paragraph=True ထည့်လိုက်ပါက စာကြောင်းလိုက်ဖတ်သဖြင့် ပိုမြန်စေပါသည်)
    results = reader.readtext(gray, allowlist='0123456789R.*xX', detail=1, paragraph=False) 
    
    # ၃။ Grid (ဇယားကွက်) တည်ဆောက်ခြင်း
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    # ၄။ ရလာသော စာသားများကို သက်ဆိုင်ရာ ဇယားကွက်ထဲ ထည့်ခြင်း
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        c = np.searchsorted(col_edges, cx) - 1
        r = np.searchsorted(row_edges, cy) - 1
        
        if 0 <= r < num_rows and 0 <= c < active_cols:
            t = text.upper().replace('X', '*')
            # စာသားအဟောင်းရှိနေလျှင် တွဲပေးရန်
            if grid_data[r][c] == "":
                grid_data[r][c] = t
            else:
                grid_data[r][c] += f" {t}"
            
    return grid_data

# --- UI ---
st.title("🎯 Lottery Pro 2026")

with st.sidebar:
    st.header("⚙️ Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3) # Default 8 cols
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=30)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("၈ တိုင်လုံး ဖတ်နေပါသည်... စက္ကန့်အနည်းငယ် စောင့်ပေးပါ"):
            data = scan_voucher_final(img, a_cols, n_rows)
            st.session_state['sheet_data'] = data

# --- EDIT & SEND TO SHEET ---
if 'sheet_data' in st.session_state:
    st.subheader("📝 Scan ရလဒ် (ပြင်ဆင်နိုင်သည်)")
    edited_data = st.data_editor(st.session_state['sheet_data'], use_container_width=True)
                    
    if st.button("🚀 Send to Google Sheet"):
        try:
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            
            creds_dict = {
                "type": info["type"],
                "project_id": info["project_id"],
                "private_key_id": info["private_key_id"],
                "private_key": info["private_key"].replace("\\n", "\n"),
                "client_email": info["client_email"],
                "client_id": info["client_id"],
                "auth_uri": info["auth_uri"],
                "token_uri": info["token_uri"],
                "auth_provider_x509_cert_url": info["auth_provider_x509_cert_url"],
                "client_x509_cert_url": info["client_x509_cert_url"]
            }
            
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            
            clean_rows = [row for row in edited_data if any(str(cell).strip() for cell in row)]
            if clean_rows:
                sh1.append_rows(clean_rows)
                st.success("✅ Google Sheet ထဲသို့ ပို့ဆောင်ပြီးပါပြီ!")
            else:
                st.warning("ပို့ရန် ဒေတာ မရှိပါ။")

        except Exception as e:
            st.error(f"Error: {str(e)}")
