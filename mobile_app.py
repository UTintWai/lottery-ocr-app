import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.title("🎯 Lottery Pro (Ditto Fill & 8-Column Precise)")

with st.sidebar:
    num_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    row_gap = st.slider("Row Gap (အတန်းညှိရန်)", 10, 50, 25)

uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 အကုန်ဖတ်မည် (Ditto စနစ်ပါဝင်သည်)"):
        with st.spinner("အကွက်မကျန်အောင် ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            results = reader.readtext(img, detail=1)
            results.sort(key=lambda x: np.mean([p[1] for p in x[0]]))

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
                        # Ditto Mark သို့မဟုတ် သင်္ကေတများကို ဖမ်းယူရန်
                        if any(c in txt for c in ['"', '။', '=', '〃', 'll']):
                            row_cells[c_idx] = "DITTO"
                        else:
                            # ဂဏန်းသန့်စင်ခြင်း
                            clean_txt = txt.upper().replace('O','0').replace('I','1').replace('S','5')
                            row_cells[c_idx] = clean_txt
                final_grid.append(row_cells)

            # --- DITTO LOGIC (အပေါ်ကတန်ဖိုး ကူးထည့်ခြင်း) ---
            for r_idx in range(len(final_grid)):
                for c_idx in range(num_cols):
                    if final_grid[r_idx][c_idx] == "DITTO" or final_grid[r_idx][c_idx] == "":
                        # အကယ်၍ အပေါ်မှာ တန်ဖိုးရှိခဲ့ရင် ကူးယူမည်
                        if r_idx > 0:
                            final_grid[r_idx][c_idx] = final_grid[r_idx-1][c_idx]

            st.session_state['ocr_data'] = final_grid

if 'ocr_data' in st.session_state:
    st.subheader(f"📊 ဖတ်ရရှိသည့် အတန်းအရေအတွက် - {len(st.session_state['ocr_data'])}")
    # ဒေတာပြင်ဆင်ရန်
    edited_data = st.data_editor(st.session_state['ocr_data'], use_container_width=True)
    
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
            sh1.append_rows(edited_data)
            
            # Sheet 2: Calculation
            master_sum = {}
            for row in edited_data:
                for i in range(0, len(row)-1, 2):
                    n = str(row[i]).strip()
                    a = str(row[i+1]).strip()
                    if n and a:
                        # ဂဏန်းမဟုတ်တာတွေဖယ်ပြီး ပေါင်းမည်
                        num_a = "".join(filter(str.isdigit, a))
                        val = int(num_a) if num_a else 0
                        master_sum[n] = master_sum.get(n, 0) + val
            
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.success("✅ Ditto များအပါအဝင် ဒေတာအားလုံး ပို့ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")