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
def load_fast_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_fast_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎯 Lottery Fix (၈ တိုင် နှင့် Sheet ပို့လွှတ်မှု ပြင်ဆင်ချက်)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("တိုင်အရေအတွက်", ["2", "4", "6", "8"], index=2) # Default 6
    num_cols = int(col_mode)
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုး)", min_value=100, value=5000)

# --- ၂။ OCR Reading Logic ---
uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 OCR ဖြင့် ဖတ်မည်"):
        with st.spinner(f"{num_cols} တိုင်စနစ်ဖြင့် ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            results = reader.readtext(img, detail=1)
            
            # Row mapping logic
            grid_data = [["" for _ in range(num_cols)] for _ in range(50)] # တန်း ၅၀ အထိ ကြိုပြင်ထားသည်
            
            col_width = w / num_cols
            row_height = h / 50 

            for (bbox, text, prob) in results:
                if prob < 0.2: continue
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                c_idx = int(cx // col_width)
                r_idx = int(cy // row_height)

                if 0 <= r_idx < 50 and 0 <= c_idx < num_cols:
                    txt = text.upper().strip()
                    # စာလုံးမှ ဂဏန်းသို့ အတင်းပြောင်းခြင်း
                    repls = {'O':'0','I':'1','S':'5','G':'6','Z':'7','B':'8','A':'4','T':'7'}
                    for k, v in repls.items(): txt = txt.replace(k, v)
                    
                    if c_idx % 2 == 0: txt = re.sub(r'[^0-9R]', '', txt) # နံပါတ်တိုင်
                    else: txt = re.sub(r'[^0-9X*]', '', txt) # ပမာဏတိုင်
                    
                    if grid_data[r_idx][c_idx]: grid_data[r_idx][c_idx] += txt
                    else: grid_data[r_idx][c_idx] = txt

            # စာကြောင်းအလွတ်များ ဖယ်ထုတ်ခြင်း
            final_rows = [row for row in grid_data if any(cell.strip() for cell in row)]
            st.session_state['ocr_res'] = final_rows

# --- ၃။ Sheet Logic (ဒေတာ ပျောက်မသွားစေရန် ပြင်ဆင်ချက်) ---
if 'ocr_res' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး လိုအပ်ပါက ပြင်ဆင်ပါ")
    edited_data = st.data_editor(st.session_state['ocr_res'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            # GCP Secrets ချိတ်ဆက်မှု
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            
            # Spreadsheet ကို အမည်ဖြင့် ဖွင့်ခြင်း
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw Data (ပို့လိုက်သော ဒေတာ အကုန်ထည့်မည်)
            sh1 = ss.get_worksheet(0) # ပထမဆုံး Tab
            sh1.append_rows(edited_data)
            
            # Sheet 2: Calculation
            master_sum = {}
            for row in edited_data:
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        amt_num = re.sub(r'\D','', a)
                        val = int(amt_num) if amt_num else 0
                        master_sum[n] = master_sum.get(n, 0) + val
            
            sh2 = ss.get_worksheet(1) # ဒုတိယ Tab
            sh2.clear()
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.balloons()
            st.success("✅ Sheets အားလုံးထဲသို့ ဒေတာများ အောင်မြင်စွာ ရောက်ရှိသွားပါပြီ!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("အကြံပြုချက်: Google Sheet ရဲ့ အမည်သည် 'LotteryData' ဖြစ်ရပါမည်။ Tab အနည်းဆုံး ၂ ခု ရှိရပါမည်။")