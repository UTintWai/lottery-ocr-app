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
    # ဖတ်နှုန်းမြန်စေရန် model ကို light ဖြစ်အောင် ထားပါသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎰 Lottery OCR (Multi-Column Fast Mode)")

# Sidebar တွင် တိုင် ၂၊ ၄၊ ၆၊ ၈ စိတ်ကြိုက်ရွေးရန်
with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("တိုင်အရေအတွက် ရွေးပါ", ["2", "4", "6", "8"], index=2) # Default 6
    num_cols = int(col_mode)
    num_rows = st.number_input("Rows (စာကြောင်းရေ)", min_value=10, value=30)
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုး)", min_value=100, value=5000)

# ---------------- ၂။ OCR Scanning Logic ----------------
uploaded_file = st.file_uploader("လက်ရေးမူပုံကို တင်ပေးပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="မူရင်းပုံ", use_container_width=True)

    if st.button("🔍 အမြန်နှုန်းဖြင့် ဖတ်မည်"):
        with st.spinner(f"{num_cols} တိုင်စနစ်ဖြင့် ခွဲခြားနေပါသည်..."):
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            # ဖတ်နှုန်းမြန်စေရန် thresholding ကို ရိုးရှင်းစွာသုံးသည်
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray, detail=1) # Fast reading mode

            col_width = w / num_cols
            row_height = h / num_rows

            for (bbox, text, prob) in results:
                if prob < 0.2: continue
                
                # စာလုံးဗဟိုမှတ်ဖြင့် အကွက်ချခြင်း
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                c_idx = int(cx // col_width)
                r_idx = int(cy // row_height)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    txt = text.upper().strip()
                    # စာလုံးအမှားပြင်ခြင်း
                    repls = {'O':'0','I':'1','S':'5','G':'6','Z':'7','B':'8','A':'4','T':'7','L':'1'}
                    for k, v in repls.items(): txt = txt.replace(k, v)
                    
                    if c_idx % 2 == 0: txt = re.sub(r'[^0-9R]', '', txt)
                    else: txt = re.sub(r'[^0-9X*]', '', txt)
                    
                    if grid_data[r_idx][c_idx]: grid_data[r_idx][c_idx] += txt
                    else: grid_data[r_idx][c_idx] = txt

            # Ditto (") Logic
            for c in range(num_cols):
                last_v = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if curr in ['"', "''", "v", "V", "11", "ll", "-"] and last_v:
                        grid_data[r][c] = last_v
                    elif curr: last_v = curr
            st.session_state['ocr_data'] = grid_data

# ---------------- ၃။ Sheet & Logic ----------------
if 'ocr_data' in st.session_state:
    final_data = st.data_editor(st.session_state['ocr_data'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            # ချိတ်ဆက်မှု အပိုင်း
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw Data (ဒေတာအကုန်ပို့သည်)
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(final_data)

            # ပေါင်းခြင်း Logic (၂ ကွက်တွဲစီ စစ်သည်)
            master_sum = {}
            for row in final_data:
                for i in range(0, len(row)-1, 2):
                    n = str(row[i]).strip()
                    a = str(row[i+1]).strip()
                    if n and a:
                        # ဂဏန်းနှင့် ပမာဏ ခွဲထုတ်ခြင်း
                        nums_only = re.sub(r'\D','',a)
                        amt = int(nums_only) if nums_only else 0
                        master_sum[n] = master_sum.get(n, 0) + amt

            # Sheet 2: Total Sum
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + [[k, v] for k, v in sorted(master_sum.items())])

            # Sheet 3: Excess (Limit ကျော်တာများ)
            sh3 = ss.get_worksheet(2)
            sh3.clear()
            excess_list = [[k, v - bet_limit] for k, v in master_sum.items() if v > bet_limit]
            if excess_list:
                sh3.append_rows([["ဂဏန်း", "ပိုလျှံငွေ"]] + sorted(excess_list))
            
            st.success("✅ Sheets 1, 2, 3 အားလုံးထဲသို့ ဒေတာများ အောင်မြင်စွာ ရောက်ရှိသွားပါပြီ!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")