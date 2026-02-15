import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- ၁။ OCR Setup ---
@st.cache_resource
def load_full_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_full_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎰 Lottery OCR (Auto-Row & Flexible Columns)")

with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("တိုင်အရေအတွက် ရွေးပါ", ["2", "4", "6", "8"], index=2)
    num_cols = int(col_mode)
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုး)", min_value=100, value=5000)

# --- ၂။ OCR Processing (ကျဲတာကို ပြင်ဆင်ထားသော စနစ်) ---
uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 OCR ဖြင့် ဖတ်မည်"):
        with st.spinner("စာကြောင်းများကို အလိုအလျောက် တွက်ချက်နေပါသည်..."):
            h, w = img.shape[:2]
            results = reader.readtext(img, detail=1)

            # ၁။ စာလုံးအားလုံးကို အမြင့် (Y coordinate) အလိုက် အရင်စီမည်
            results.sort(key=lambda x: np.mean([p[1] for p in x[0]]))

            rows = []
            if results:
                current_row = [results[0]]
                # ၂။ အမြင့်ချင်း နီးစပ်တာတွေကို တစ်ကြောင်းတည်းအဖြစ် အုပ်စုဖွဲ့မည်
                for i in range(1, len(results)):
                    prev_y = np.mean([p[1] for p in current_row[-1][0]])
                    curr_y = np.mean([p[1] for p in results[i][0]])
                    
                    # စာကြောင်းအမြင့် ကွာခြားချက် (ပုံစံအမျိုးမျိုးအတွက် ညှိထားသည်)
                    if abs(curr_y - prev_y) < (h / 45): 
                        current_row.append(results[i])
                    else:
                        rows.append(current_row)
                        current_row = [results[i]]
                rows.append(current_row)

            # ၃။ တစ်ကြောင်းချင်းစီအတွင်းမှာ ဘယ်ကနေ ညာသို့ (X coordinate) စီပြီး Grid ထဲထည့်မည်
            final_data = []
            for r in rows:
                r.sort(key=lambda x: np.mean([p[0] for p in x[0]]))
                row_cells = ["" for _ in range(num_cols)]
                
                for item in r:
                    cx = np.mean([p[0] for p in item[0]])
                    c_idx = int(cx // (w / num_cols))
                    if 0 <= c_idx < num_cols:
                        txt = item[1].upper().strip()
                        # Clean Text
                        txt = txt.replace('O','0').replace('I','1').replace('S','5').replace('G','6').replace('Z','7')
                        if c_idx % 2 == 0: txt = re.sub(r'[^0-9R]', '', txt)
                        else: txt = re.sub(r'[^0-9X*]', '', txt)
                        
                        if row_cells[c_idx]: row_cells[c_idx] += txt
                        else: row_cells[c_idx] = txt
                final_data.append(row_cells)

            st.session_state['ocr_final'] = final_data

# --- ၃။ Editing & Google Sheet Upload ---
if 'ocr_final' in st.session_state:
    st.subheader(f"📝 {num_cols} တိုင် ရလဒ် (စာကြောင်း {len(st.session_state['ocr_final'])} ကြောင်း)")
    edited_data = st.data_editor(st.session_state['ocr_final'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw Data
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_data)
            
            # Sheet 2: Calculation (၂ ကွက်တွဲစီ ပေါင်းခြင်း)
            master_sum = {}
            for row in edited_data:
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        clean_a = re.sub(r'\D','', a)
                        val = int(clean_a) if clean_a else 0
                        master_sum[n] = master_sum.get(n, 0) + val
            
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.success("✅ ဒေတာများ အောင်မြင်စွာ ပို့ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")