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
    # EasyOCR ကို CPU တွင် မြန်ဆန်အောင် ဆက်တင်ညှိထားခြင်း
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- SCAN FUNCTION (၈ တိုင်လုံး တိကျစွာဖတ်ရန်) ---
def scan_voucher_final(img, active_cols, num_rows):
    # ၁။ ပုံကို ၅၀% ချုံ့ခြင်းဖြင့် OCR မြန်နှုန်းကို တိုးမြှင့်သည်
    img_resized = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ပုံ၏ အမြင့်နှင့် အနံကို အတိအကျရယူခြင်း
    h, w = gray.shape

    # ၂။ OCR ဖတ်ခြင်း (စာသားနှင့် တည်နေရာကို ယူသည်)
    results = reader.readtext(gray, allowlist='0123456789R.*xX', detail=1) 
    
    # ၃။ Grid (ဇယားကွက်) တည်ဆောက်ခြင်း
    # ပုံ၏ အနံ (w) ကို အတိုင်အရေအတွက်အလိုက် အညီအမျှ ခွဲဝေသည်
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    # ၄။ ရလာသော စာသားများကို သက်ဆိုင်ရာ အတိုင်/အတန်းထဲ ထည့်ခြင်း
    for (bbox, text, prob) in results:
        # စာလုံး၏ ဗဟိုအမှတ်ကို ရှာဖွေခြင်း
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        # မည်သည့် အတိုင် (Column) နှင့် အတန်း (Row) ထဲကျသလဲ တွက်ချက်ခြင်း
        c = np.searchsorted(col_edges, cx) - 1
        r = np.searchsorted(row_edges, cy) - 1
        
        # Grid ဘောင်အတွင်းရှိပါက ဇယားထဲ ထည့်သွင်းသည်
        if 0 <= r < num_rows and 0 <= c < active_cols:
            t = text.upper().replace('X', '*')
            # အကယ်၍ အကွက်ထဲမှာ စာရှိနှင့်ပြီးသားဖြစ်ပါက ကော်မာ (,) ဖြင့် တွဲပေးမည်
            if grid_data[r][c] == "":
                grid_data[r][c] = t
            else:
                grid_data[r][c] += f", {t}"
            
    return grid_data

# --- UI ---
st.title("🎯 Lottery Pro 2026")

with st.sidebar:
    st.header("⚙️ Settings")
    # အတိုင်အရေအတွက်ကို ၈ တိုင်အထိ ရွေးနိုင်သည်
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=30)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("ဖတ်နေပါသည်... အတိုင်များလေ ပိုကြာလေဖြစ်ပါသည်"):
            data = scan_voucher_final(img, a_cols, n_rows)
            st.session_state['sheet_data'] = data

# --- EDIT & SEND TO SHEET ---
if 'sheet_data' in st.session_state:
    st.subheader("📝 Edit Data (မပို့မီ လိုအပ်သည်များ ပြင်ဆင်ပါ)")
    edited_data = st.data_editor(st.session_state['sheet_data'], use_container_width=True)
                    
    if st.button("🚀 Send to Google Sheet"):
        try:
            # Secrets ဖတ်ခြင်း (အမည်ကို Dashboard တွင် အတိအကျတူအောင် ပေးထားပါ)
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
            
            # စာသားရှိသော Row များကိုသာ စစ်ထုတ်ပို့ဆောင်ခြင်း
            clean_rows = [row for row in edited_data if any(str(cell).strip() for cell in row)]
            
            if clean_rows:
                sh1.append_rows(clean_rows)
                st.success("✅ ဒေတာများကို Google Sheet ထဲသို့ ပို့ပြီးပါပြီ!")
            else:
                st.warning("ပို့ရန် ဒေတာ မရှိပါ။")

        except Exception as e:
            st.error(f"Error တက်နေပါသည်: {str(e)}")
