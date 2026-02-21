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
    # လက်ရေးနှင့် သုံးလုံးဂဏန်းများအတွက် ပိုမိုကောင်းမွန်အောင် ထားရှိသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def scan_voucher_final(img, active_cols, num_rows):
    # ၁။ ပုံကို OCR ဖတ်ရလွယ်ကူအောင် ပြင်ဆင်ခြင်း
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # အလင်းအမှောင်နှင့် စာလုံးများကို ပိုမိုထင်ရှားစေခြင်း
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    h, w = gray.shape

    # ၂။ OCR ဖတ်ခြင်း (ဂဏန်းနှင့် သင်္ကေတများသာ ခွင့်ပြုမည်)
    results = reader.readtext(gray, allowlist='0123456789R.*xX" ', detail=1) 
    
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
            t = text.upper().replace('X', '*').strip()
            # Ditto အမှတ်အသားများကို ခွဲခြားရန်
            if t in ['"', '||', '။', '..']:
                t = "DITTO"
            
            if grid_data[r][c] == "":
                grid_data[r][c] = t
            else:
                grid_data[r][c] += f" {t}"
    
    # ၃။ DITTO LOGIC (အပေါ်ကတန်ဖိုးကို အလိုအလျောက် ကူးဖြည့်ပေးခြင်း)
    for c in range(active_cols):
        for r in range(1, num_rows):
            if grid_data[r][c] == "DITTO" or grid_data[r][c] == "":
                # အကယ်၍ အပေါ်ကွက်မှာ တန်ဖိုးရှိနေလျှင် ယူသုံးမည်
                # (သို့သော် ဂဏန်းမဟုတ်သော အကွက်လွတ်များကို မဖြည့်မိစေရန် စစ်ဆေးပါ)
                if grid_data[r-1][c] != "":
                    # လက်တွေ့တွင် အကွက်လွတ်တိုင်း မဖြည့်စေရန် Ditto ရှိမှသာ ဖြည့်ခြင်းက ပိုစိတ်ချရသည်
                    if grid_data[r][c] == "DITTO":
                        grid_data[r][c] = grid_data[r-1][c]
                
    return grid_data

# --- UI ---
st.title("🎯 Lottery Pro 2026")

with st.sidebar:
    st.header("⚙️ Settings")
    # ၈ တိုင်ဖတ်ရန် ၈ ကို ရွေးပေးပါ
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=35)
    sheet_option = st.radio("ဒေတာပို့မည့် Sheet", ["Sheet1", "Sheet2", "Sheet3"])

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True, caption="တင်ထားသောပုံ")

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("၈ တိုင်လုံးကို အပြည့်အစုံ ဖတ်နေပါသည်..."):
            data = scan_voucher_final(img, a_cols, n_rows)
            st.session_state['sheet_data'] = data

if 'sheet_data' in st.session_state:
    st.subheader(f"📝 {sheet_option} အတွက် Scan ရလဒ်")
    # အမှားများကို ဤနေရာတွင် ကိုယ်တိုင်ပြင်ဆင်နိုင်ပါသည်
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
            sh = ss.worksheet(sheet_option)
            
            clean_rows = [row for row in edited_data if any(str(cell).strip() for cell in row)]
            if clean_rows:
                sh.append_rows(clean_rows)
                st.success(f"✅ {sheet_option} ထဲသို့ ဒေတာများ ပို့ဆောင်ပြီးပါပြီ!")
            else:
                st.warning("ပို့ရန် ဒေတာ မတွေ့ပါ။")

        except Exception as e:
            st.error(f"Error: {str(e)}")
