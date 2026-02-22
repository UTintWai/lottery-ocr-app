import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026 Ultimate", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def clean_ocr_text(txt):
    txt = txt.upper().strip()
    repls = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'7','T':'7','Q':'0','D':'0'}
    for k,v in repls.items():
        txt = txt.replace(k,v)
    return txt

def is_ditto(txt):
    ditto_marks = ['"', '။', '=', '||', '..', '`', '“', '4', 'U'] # OCR မှားတတ်သော Ditto ပုံစံများ
    return any(mark in txt for mark in ditto_marks)

# --- UI ---
with st.sidebar:
    st.header("⚙️ Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    row_tolerance = st.slider("Row Sensitivity", 5, 50, 20) # အတန်းညှိရန်

uploaded_file = st.file_uploader("လက်ရေး Voucher ပုံတင်ပါ", type=["jpg","jpeg","png"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    
    if st.button("🔍 Scan & Fix Grid"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        results = reader.readtext(gray, detail=1)

        # ၁။ စာလုံးများကို အမြင့် (Y-coordinate) အလိုက် အတန်းခွဲခြင်း
        data_list = []
        for (bbox, text, prob) in results:
            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox])
            data_list.append({'x': cx, 'y': cy, 'text': text})

        # အတန်းလိုက် စုစည်းခြင်း Logic
        data_list.sort(key=lambda x: x['y'])
        rows = []
        if data_list:
            current_row = [data_list[0]]
            for i in range(1, len(data_list)):
                if data_list[i]['y'] - current_row[-1]['y'] < row_tolerance:
                    current_row.append(data_list[i])
                else:
                    rows.append(sorted(current_row, key=lambda x: x['x']))
                    current_row = [data_list[i]]
            rows.append(sorted(current_row, key=lambda x: x['x']))

        # ၂။ ဇယားကွက်ထဲသို့ ပြန်ထည့်ခြင်း
        final_grid = []
        for r in rows:
            grid_row = ["" for _ in range(a_cols)]
            col_width = w / a_cols
            for item in r:
                c_idx = int(item['x'] / col_width)
                if 0 <= c_idx < a_cols:
                    txt = clean_ocr_text(item['text'])
                    if is_ditto(txt):
                        txt = "DITTO"
                    else:
                        # ၃ လုံးဂဏန်း (0) ဖြည့်ခြင်း
                        nums = re.findall(r'\d+', txt)
                        if nums:
                            val = nums[0]
                            if len(val) < 3 and '*' not in txt: 
                                val = val.zfill(3)
                            txt = val
                    grid_row[c_idx] = txt
            final_grid.append(grid_row)

        # ၃။ Ditto Auto-fill
        for c in range(a_cols):
            for r in range(1, len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = final_grid[r-1][c]

        st.session_state['data_final'] = final_grid

if 'data_final' in st.session_state:
    st.subheader("📝 Scan ရလဒ် (အကွက်ကျကျ ပြင်ဆင်ပြီး)")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)
    
    if st.button("🚀 Send to Google Sheets"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            sh = ss.worksheet("Sheet1")
            
            # ရှေ့က 0 မပျောက်ရန် ' ခံပြီး ပို့ခြင်း
            formatted_data = [[f"'{cell}" if cell != "" else "" for cell in row] for row in edited_data]
            sh.append_rows(formatted_data)
            st.success("✅ ဒေတာများကို Sheet ထဲသို့ ပို့ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
