import streamlit as st
import numpy as np
import easyocr
import cv2
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro 2026 Ultimate", layout="wide")

@st.cache_resource
def load_ocr():
    # သုံးလုံးဂဏန်းနှင့် ထိုးကြေးများ အပြည့်အစုံဖတ်နိုင်ရန် model ကို optimize လုပ်ထားသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def scan_voucher_final(img, active_cols, num_rows):
    # ၁။ ပုံကို OCR ဖတ်ရလွယ်အောင် Contrast မြှင့်တင်ခြင်း
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # ၈ တိုင်လုံး အပြည့်ပေါ်ရန် ပုံအရွယ်အစားကို အရမ်းမချုံ့တော့ဘဲ ပုံမှန်အတိုင်းထားပါမည်
    h, w = gray.shape

    # ၂။ OCR ဖတ်ခြင်း (paragraph=True က စာလုံးများကို အပြည့်အစုံဖတ်ရန် ကူညီပေးသည်)
    results = reader.readtext(gray, allowlist='0123456789R.*xX', detail=1, paragraph=False) 
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    for (bbox, text, prob) in results:
        # စာလုံး၏ ဗဟိုအမှတ်ကို ရှာဖွေခြင်း
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        c = np.searchsorted(col_edges, cx) - 1
        r = np.searchsorted(row_edges, cy) - 1
        
        if 0 <= r < num_rows and 0 <= c < active_cols:
            # စာလုံးများ အပြည့်အစုံပေါ်စေရန် Clean လုပ်ခြင်း
            t = text.upper().replace('X', '*').strip()
            if grid_data[r][c] == "":
                grid_data[r][c] = t
            else:
                # ဂဏန်းနှစ်ခု ပူးနေပါက ခွဲပေးရန်
                grid_data[r][c] += f" {t}"
            
    return grid_data

# --- UI ---
st.title("🎯 Lottery Pro 2026")

with st.sidebar:
    st.header("⚙️ Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3) # ၈ တိုင်ကို default ထားသည်
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=35)
    sheet_name = st.text_input("Sheet နာမည်", value="Sheet1") # Sheet2, Sheet3 သို့ ပို့လိုပါက ပြောင်းရန်

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("၈ တိုင်လုံး ဖတ်နေပါသည်... စက္ကန့် ၃၀ ခန့် စောင့်ပေးပါ"):
            data = scan_voucher_final(img, a_cols, n_rows)
            st.session_state['sheet_data'] = data

# --- EDIT & SEND TO SHEET ---
if 'sheet_data' in st.session_state:
    st.subheader("📝 Edit Data (Sheet ထဲ မပို့မီ လိုအပ်သည်များ ပြင်ပါ)")
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
            
            # ဒေတာပို့မည့် Sheet ကို ရွေးချယ်ခြင်း
            ss = client.open("LotteryData")
            try:
                sh = ss.worksheet(sheet_name)
            except:
                st.error(f"'{sheet_name}' ဆိုသည့် Worksheet ကို မတွေ့ပါ။ နာမည်မှန်မမှန် စစ်ပေးပါ။")
                st.stop()
            
            clean_rows = [row for row in edited_data if any(str(cell).strip() for cell in row)]
            if clean_rows:
                sh.append_rows(clean_rows)
                st.success(f"✅ ဒေတာများကို {sheet_name} ထဲသို့ ပို့ပြီးပါပြီ!")
            else:
                st.warning("ပို့ရန် ဒေတာ မရှိပါ။")

        except Exception as e:
            st.error(f"Error: {str(e)}")
