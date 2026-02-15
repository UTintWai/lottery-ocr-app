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
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎯 Lottery High-Precision (Ditto Fill System)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Configuration")
    num_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    
    # Slider အစား +/- ခလုတ် (Number Input) ဖြင့် ပြောင်းလဲခြင်း
    row_gap = st.number_input("Row Gap (အတန်းအကွာအဝေးညှိရန်)", min_value=5, max_value=100, value=25, step=1)
    
    st.info("💡 အတန်းတွေ ကျော်နေရင် Gap ကို လျှော့ပါ။ အတန်းတွေ ပေါင်းနေရင် Gap ကို တိုးပါ။")

# --- ၂။ OCR & Ditto Logic ---
uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 အကုန်ဖတ်မည်"):
        with st.spinner("Ditto logic ဖြင့် အကွက်စုံအောင် ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            results = reader.readtext(img, detail=1)
            
            # Y (အမြင့်) အလိုက် အရင်စီသည်
            results.sort(key=lambda x: np.mean([p[1] for p in x[0]]))

            # အတန်းခွဲခြားခြင်း (Clustering)
            rows = []
            if results:
                current_row = [results[0]]
                for i in range(1, len(results)):
                    prev_y = np.mean([p[1] for p in current_row[-1][0]])
                    curr_y = np.mean([p[1] for p in results[i][0]])
                    
                    if abs(curr_y - prev_y) < row_gap:
                        current_row.append(results[i])
                    else:
                        rows.append(current_row)
                        current_row = [results[i]]
                rows.append(current_row)

            # Grid ဆောက်ခြင်း
            final_grid = []
            col_width = w / num_cols
            
            for r in rows:
                r.sort(key=lambda x: np.mean([p[0] for p in x[0]]))
                row_cells = ["" for _ in range(num_cols)]
                
                for item in r:
                    cx = np.mean([p[0] for p in item[0]])
                    c_idx = int(cx // col_width)
                    if 0 <= c_idx < num_cols:
                        txt = item[1].strip()
                        # Ditto Mark ဖမ်းယူခြင်း (။ သို့မဟုတ် 〃 သို့မဟုတ် စာလုံးအသေးများ)
                        if any(char in txt for char in ['။', '"', '〃', '=', 'll', 'v', 'V']):
                            row_cells[c_idx] = "DITTO"
                        else:
                            clean_txt = txt.upper().replace('O','0').replace('I','1').replace('S','5')
                            row_cells[c_idx] = clean_txt
                final_grid.append(row_cells)

            # --- DITTO AUTO-FILL (အပေါ်ကတန်ဖိုး ကူးထည့်ခြင်း) ---
            for r_idx in range(len(final_grid)):
                for c_idx in range(num_cols):
                    if final_grid[r_idx][c_idx] == "DITTO" or final_grid[r_idx][c_idx] == "":
                        if r_idx > 0:
                            # အပေါ်အတန်းမှ တန်ဖိုးကို ယူသည်
                            final_grid[r_idx][c_idx] = final_grid[r_idx-1][c_idx]

            st.session_state['ocr_stable'] = final_grid

# --- ၃။ Table & Sheet Upload ---
if 'ocr_stable' in st.session_state:
    st.subheader(f"📊 စုစုပေါင်း အတန်းအရေအတွက်: {len(st.session_state['ocr_stable'])}")
    edited_df = st.data_editor(st.session_state['ocr_stable'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_df)
            
            # Sheet 2: Calculation
            master_sum = {}
            for row in edited_df:
                for i in range(0, len(row)-1, 2):
                    n = str(row[i]).strip()
                    a = str(row[i+1]).strip()
                    if n and a:
                        num_a = "".join(filter(str.isdigit, a))
                        val = int(num_a) if num_a else 0
                        master_sum[n] = master_sum.get(n, 0) + val
            
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.balloons()
            st.success("✅ Ditto များအပါအဝင် ဒေတာအားလုံး Sheet ထဲသို့ ရောက်သွားပါပြီ!")
        except Exception as e:
            st.error(f"❌ Sheet Error: {str(e)}")