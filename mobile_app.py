import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- ၁။ OCR & Config ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎰 Lottery OCR (6/8 Columns Stable)")

# Sidebar တွင် တိုင်အရေအတွက် ပြန်ရွေးခိုင်းခြင်း
with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("တိုင်အရေအတွက် ရွေးပါ", ["6", "8"], index=0)
    num_cols = int(col_mode)
    num_rows = st.number_input("Rows (စာကြောင်းရေ)", min_value=10, value=25)
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုးသတ်မှတ်ရန်)", min_value=100, value=5000)

# ---------------- ၂။ OCR Scanning Logic ----------------
uploaded_file = st.file_uploader("လက်ရေးမူပုံကို တင်ပေးပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="တင်ထားသောပုံ", use_container_width=True)

    if st.button("🔍 OCR ဖြင့် ဖတ်မည်"):
        with st.spinner(f"{num_cols} တိုင်စနစ်ဖြင့် ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            # OCR ဖတ်ခြင်း (Contrast မြှင့်ထားသည်)
            results = reader.readtext(img, contrast_ths=0.05, low_text=0.1)

            # Column များကို ရွေးထားသော တိုင်အရေအတွက်အလိုက် ပိုင်းခြင်း
            col_width = w / num_cols
            row_height = h / num_rows

            for (bbox, text, prob) in results:
                if prob < 0.15: continue
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                c_idx = int(cx // col_width)
                r_idx = int(cy // row_height)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    txt = text.upper().strip()
                    # စာလုံးမှ ဂဏန်းသို့ ပြောင်းလဲခြင်း
                    repls = {'O':'0','I':'1','S':'5','G':'6','Z':'7','B':'8','A':'4','T':'7','L':'1'}
                    for k, v in repls.items(): txt = txt.replace(k, v)
                    
                    if c_idx % 2 == 0: txt = re.sub(r'[^0-9R]', '', txt)
                    else: txt = re.sub(r'[^0-9X*]', '', txt)
                    
                    if grid_data[r_idx][c_idx]: grid_data[r_idx][c_idx] += txt
                    else: grid_data[r_idx][c_idx] = txt

            # Ditto Logic
            for c in range(num_cols):
                last_v = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if curr in ['"', "''", "v", "V", "11", "ll", "-", "Y"] and last_v:
                        grid_data[r][c] = last_v
                    elif curr: last_v = curr
            st.session_state['ocr_data'] = grid_data

# ---------------- ၃။ Sheet ပို့ခြင်း Logic ----------------
if 'ocr_data' in st.session_state:
    final_data = st.data_editor(st.session_state['ocr_data'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            # GCP Service Account ချိတ်ဆက်မှု
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            
            # Spreadsheet နာမည် "LotteryData" ဟု အသေထားပါသည်
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw Data
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(final_data)

            # Sheet 2 & 3 အတွက် ပေါင်းခြင်း Logic
            master_sum = {}
            for row in final_data:
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        # ဤနေရာတွင် ရိုးရှင်းသော ပမာဏပေါင်းခြင်း လုပ်ပါသည်
                        amt = int(re.sub(r'\D','',a)) if re.sub(r'\D','',a) else 0
                        master_sum[n] = master_sum.get(n, 0) + amt

            # Sheet 2: Summing
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["Number", "Total"]] + [[k, v] for k, v in master_sum.items()])

            # Sheet 3: ပိုလျှံတန်ဖိုး (Limit ကျော်တာများ)
            sh3 = ss.get_worksheet(2)
            sh3.clear()
            excess = [[k, v - bet_limit] for k, v in master_sum.items() if v > bet_limit]
            if excess:
                sh3.append_rows([["ဂဏန်း", "ပိုလျှံငွေ"]] + excess)
            
            st.success("✅ Sheets 1, 2, 3 အားလုံးထဲသို့ ဒေတာများ ရောက်ရှိသွားပါပြီ!")
            
        except Exception as e:
            st.error(f"❌ Sheet Error: {str(e)}")