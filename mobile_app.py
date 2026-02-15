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
def load_optimized_ocr():
    # ဖတ်နှုန်းမြန်စေရန် GPU မပါဘဲ အကောင်းဆုံးချိန်ညှိထားသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_optimized_ocr()

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")
st.title("🎰 Lottery OCR (၈ တိုင် တိကျဖတ်ရှုမှု စနစ်)")

with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("တိုင်အရေအတွက်", ["2", "4", "6", "8"], index=3)
    num_cols = int(col_mode)
    # Row sensitivity ကို ၂၅ တန်းအတွက် ၂၀ ဝန်းကျင်ထားရန် အကြံပြုသည်
    row_gap = st.slider("Row Gap (အတန်းခွဲခြားမှု)", 10, 50, 20)
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုး)", min_value=100, value=5000)

# --- ၂။ OCR Processing (အကွက်မကျန်စေရန် ပြင်ဆင်ချက်) ---
uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 အမြန်နှုန်းဖြင့် အကုန်ဖတ်မည်"):
        with st.spinner("၂၅ တန်းစလုံးကို အကွက်မကျန်အောင် ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            # contrast မြှင့်တင်ခြင်းဖြင့် ဖတ်ရပိုလွယ်အောင်လုပ်သည်
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray, detail=1, paragraph=False)

            # အမြင့် (Y) အလိုက် sorting လုပ်သည်
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

            # ဒေတာများကို Grid ထဲ ထည့်သွင်းခြင်း
            final_data = []
            col_width = w / num_cols
            
            for r in rows:
                r.sort(key=lambda x: np.mean([p[0] for p in x[0]]))
                row_cells = ["" for _ in range(num_cols)]
                
                for item in r:
                    cx = np.mean([p[0] for p in item[0]])
                    c_idx = int(cx // col_width)
                    
                    if 0 <= c_idx < num_cols:
                        txt = item[1].upper().strip()
                        # Character Repair
                        txt = txt.replace('O','0').replace('S','5').replace('I','1').replace('Z','7').replace('B','8').replace('G','6')
                        
                        # ဂဏန်းတိုင်နှင့် ပမာဏတိုင် ခွဲခြားသန့်စင်ခြင်း
                        if c_idx % 2 == 0:
                            txt = re.sub(r'[^0-9R]', '', txt)
                        else:
                            txt = re.sub(r'[^0-9X*]', '', txt)
                        
                        if row_cells[c_idx]: row_cells[c_idx] += txt
                        else: row_cells[c_idx] = txt
                
                if any(row_cells):
                    final_data.append(row_cells)

            st.session_state['ocr_result'] = final_data

# --- ၃။ Editing & Sheet Upload (ပျောက်မသွားစေရန် တိုက်ရိုက်ပို့ခြင်း) ---
if 'ocr_result' in st.session_state:
    st.subheader(f"📝 စုစုပေါင်း {len(st.session_state['ocr_result'])} တန်း ဖတ်ရှိရပါသည်")
    edited_df = st.data_editor(st.session_state['ocr_result'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ အကုန်ပို့မည်"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Raw Data
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_df)
            
            # Sheet 2: Master Sum (၂ ကွက်တွဲစီ စစ်ဆေးသည်)
            master_sum = {}
            for row in edited_df:
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        amt_clean = "".join(filter(str.isdigit, a))
                        val = int(amt_clean) if amt_clean else 0
                        master_sum[n] = master_sum.get(n, 0) + val
            
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + [[k, v] for k, v in sorted(master_sum.items())])
            
            st.success("✅ အကုန်လုံးဖတ်ပြီး Sheet ထဲသို့ ဒေတာများ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")