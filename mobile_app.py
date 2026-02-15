import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- ၁။ OCR Setup (Fast Mode) ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎯 Lottery Fast-Scan (2, 4, 6, 8 တိုင်)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("တိုင်အရေအတွက်", ["2", "4", "6", "8"], index=2)
    num_cols = int(col_mode)
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုး)", min_value=100, value=5000)

# --- ၂။ OCR Function (New Logic) ---
uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 ဖတ်မည် (Fast Mode)"):
        with st.spinner("စာလုံးများကို တည်နေရာအလိုက် စီနေပါသည်..."):
            # OCR ဖတ်ခြင်း (detail=1 ကို သုံးမှ တည်နေရာရမည်)
            results = reader.readtext(img, detail=1, paragraph=False)
            
            # ၁။ စာလုံးအားလုံးကို အမြင့် (Y coordinate) အလိုက် အရင်စီမည်
            results.sort(key=lambda x: x[0][0][1]) 

            rows = []
            if results:
                current_row = [results[0]]
                # ၂။ စာကြောင်းတူတာတွေကို အုပ်စုဖွဲ့မည် (အမြင့်ချင်း နီးစပ်တာကို တစ်ကြောင်းတည်းထား)
                for i in range(1, len(results)):
                    prev_y = np.mean([p[1] for p in current_row[-1][0]])
                    curr_y = np.mean([p[1] for p in results[i][0]])
                    
                    if abs(curr_y - prev_y) < 25: # စာကြောင်းအမြင့် ကွာခြားချက် limit
                        current_row.append(results[i])
                    else:
                        rows.append(current_row)
                        current_row = [results[i]]
                rows.append(current_row)

            # ၃။ တစ်ကြောင်းချင်းစီအတွင်းမှာ ဘယ်ကနေ ညာသို့ (X coordinate) ပြန်စီမည်
            final_grid = []
            for r in rows:
                r.sort(key=lambda x: x[0][0][0])
                row_data = ["" for _ in range(num_cols)]
                
                # ပုံ၏ အကျယ်ကို တိုင်အရေအတွက်ဖြင့် စား၍ နေရာချမည်
                img_w = img.shape[1]
                for item in r:
                    cx = np.mean([p[0] for p in item[0]])
                    c_idx = int(cx // (img_w / num_cols))
                    if c_idx < num_cols:
                        txt = item[1].upper().replace('O','0').replace('S','5').replace('I','1')
                        row_data[c_idx] = txt
                final_grid.append(row_data)

            st.session_state['ocr_res'] = final_grid

# --- ၃။ Editing & Sheet Upload ---
if 'ocr_res' in st.session_state:
    edited_df = st.data_editor(st.session_state['ocr_res'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            # Credentials & Connection
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw (Overwrite mode for speed)
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_df)

            # Sheet 2: Calculation
            master = {}
            for row in edited_df:
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        clean_a = re.sub(r'\D','', a)
                        val = int(clean_a) if clean_a else 0
                        master[n] = master.get(n, 0) + val

            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["Number", "Total"]] + [[k, v] for k, v in master.items()])
            
            st.success("✅ Google Sheet ထဲသို့ ဒေတာများ ရောက်ရှိသွားပါပြီ!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")