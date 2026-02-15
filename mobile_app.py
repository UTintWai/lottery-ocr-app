import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

@st.cache_resource
def load_ocr_model():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr_model()

st.title("🎯 Lottery High-Precision (၈ တိုင် အပြည့်အစုံ)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    col_mode = st.selectbox("တိုင်အရေအတွက်", ["2", "4", "6", "8"], index=3)
    num_cols = int(col_mode)
    row_sensitivity = st.slider("Row Sensitivity (စာကြောင်း ခွဲခြားမှု)", 10, 40, 20)

uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 အသေးစိတ် စစ်ဆေးဖတ်ရှုမည်"):
        with st.spinner("စာကြောင်း ၂၅ ကြောင်းစလုံးကို အတိအကျ ရှာဖွေနေပါသည်..."):
            h, w = img.shape[:2]
            # OCR results
            results = reader.readtext(img, detail=1, paragraph=False)
            
            # ၁။ အမြင့် (Y) အလိုက် အုပ်စုဖွဲ့ စီစဥ်ခြင်း
            results.sort(key=lambda x: np.mean([p[1] for p in x[0]]))
            
            rows = []
            if results:
                current_row = [results[0]]
                for i in range(1, len(results)):
                    prev_y = np.mean([p[1] for p in current_row[-1][0]])
                    curr_y = np.mean([p[1] for p in results[i][0]])
                    
                    # Row spacing ကို sensitivity အလိုက် ညှိခြင်း
                    if abs(curr_y - prev_y) < row_sensitivity:
                        current_row.append(results[i])
                    else:
                        rows.append(current_row)
                        current_row = [results[i]]
                rows.append(current_row)

            # ၂။ ဒေတာများကို Grid ထဲ ထည့်သွင်းခြင်း
            final_data = []
            for r in rows:
                r.sort(key=lambda x: np.mean([p[0] for p in x[0]]))
                row_cells = ["" for _ in range(num_cols)]
                
                for item in r:
                    cx = np.mean([p[0] for p in item[0]])
                    # Column calculation with strict boundary
                    c_idx = int(cx // (w / num_cols))
                    
                    if 0 <= c_idx < num_cols:
                        txt = item[1].upper().strip()
                        # Character cleaning
                        txt = txt.replace('O','0').replace('I','1').replace('S','5').replace('G','6').replace('Z','7').replace('B','8')
                        
                        # Column Type check
                        if c_idx % 2 == 0:
                            txt = re.sub(r'[^0-9R]', '', txt)
                        else:
                            txt = re.sub(r'[^0-9X*]', '', txt)
                        
                        if row_cells[c_idx]: row_cells[c_idx] += f" {txt}"
                        else: row_cells[c_idx] = txt
                
                # အကယ်၍ row ထဲမှာ ဘာစာမှမရှိရင် မထည့်ပါ
                if any(row_cells):
                    final_data.append(row_cells)

            st.session_state['ocr_stable'] = final_data

# --- Display & Google Sheet ---
if 'ocr_stable' in st.session_state:
    st.subheader(f"📊 ရရှိလာသော အတန်းအရေအတွက်: {len(st.session_state['ocr_stable'])}")
    edited_df = st.data_editor(st.session_state['ocr_stable'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ အကုန်ပို့မည်"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1 (Raw)
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_df)
            
            # Sheet 2 (Summing)
            master_sum = {}
            for row in edited_df:
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        # Extract only digits for sum
                        amt_val = "".join(filter(str.isdigit, a))
                        val = int(amt_val) if amt_val else 0
                        master_sum[n] = master_sum.get(n, 0) + val
            
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["Number", "Total"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.success("✅ အားလုံးဖတ်ပြီး Sheet ထဲသို့ တိကျစွာ ပို့ဆောင်ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"Error: {str(e)}")