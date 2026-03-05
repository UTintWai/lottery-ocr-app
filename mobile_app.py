import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v82", layout="wide")
SHEET_NAME = "LotteryData" 

def save_to_gsheet(data_to_save):
    try:
        if not data_to_save: return False
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).get_worksheet(0)
        formatted_rows = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data_to_save]
        sheet.append_rows(formatted_rows, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_v82(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    img = cv2.resize(img, (1600, int(h * (1600 / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ပုံကို ပိုမိုပြတ်သားအောင် လုပ်ဆောင်ခြင်း
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    results = reader.readtext(enhanced, paragraph=False)
    
    # 🎯 ၂၅ တန်း အတိအကျ ပေတံ (Bins)
    bins = np.linspace(30, img.shape[0]-30, 26) 
    grid = [["" for _ in range(4)] for _ in range(25)]
    
    col_w = 1600 // 4
    for (bbox, text, prob) in results:
        # စာလုံးရဲ့ အလယ်ဗဟိုကို တွက်ချက်ခြင်း
        x_c = np.mean([p[0] for p in bbox])
        y_c = np.mean([p[1] for p in bbox])
        
        c_idx = int(x_c // col_w)
        if c_idx > 3: c_idx = 3
        
        t = text.upper().replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
        
        # ၃ လုံးဂဏန်းတိုင်များအတွက် (Column 0, 2)
        if c_idx % 2 == 0: 
            val = re.sub(r'[^0-9]', '', t)
            if val: val = val.zfill(3)[-3:]
        # ထိုးကြေးတိုင်များအတွက် (Column 1, 3)
        else:
            if re.search(r'[။၊"=“_…\.\-\']', t):
                val = "DITTO"
            else:
                val = re.sub(r'[^0-9]', '', t)

        # 🎯 SMART SNAP: နီးစပ်ရာ Bin ထဲသို့ ထည့်ခြင်း
        b_idx = np.digitize(y_c, bins) - 1
        if 0 <= b_idx < 25:
            # အကယ်၍ အကွက်ထဲမှာ ရှိနှင့်နေပြီးသားဆိုလျှင် Duplicate မဖြစ်အောင် စစ်ဆေးသည်
            if grid[b_idx][c_idx] == "":
                grid[b_idx][c_idx] = val
            elif val != "DITTO": # ဂဏန်းအသစ်တွေ့လျှင် အစားထိုးသည်
                grid[b_idx][c_idx] = val

    # Ditto Filling
    for c in [1, 3]:
        last = ""
        for r in range(25):
            v = grid[r][c].strip()
            if v.isdigit() and v != "": 
                last = v
            elif (v == "DITTO" or v == "") and last != "":
                grid[r][c] = last
                
    return grid

# --- UI ---
st.title("🔢 Lottery Pro v82 (Micro-Snap Mode)")
st.info("အသေးစား အတန်းလွဲချော်မှုများကို အလိုအလျောက် ပြန်လည်ညှိပေးပါမည်။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_r = st.file_uploader("ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 Run Precision Scan"):
    with st.spinner('အတန်းများကို ချိန်ညှိနေပါသည်...'):
        l_res = process_v82(cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), 1)) if up_l else [[""]*4]*25
        r_res = process_v82(cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), 1)) if up_r else [[""]*4]*25
        
        # ဘယ်ညာ တိုက်ရိုက်ပေါင်းစပ်ခြင်း
        final = [l_res[i] + r_res[i] for i in range(25)]
        st.session_state['data_v82'] = final
        st.rerun()

if 'data_v82' in st.session_state:
    st.subheader("📝 စစ်ဆေးရန် (အမှားရှိက ပြင်နိုင်သည်)")
    # ၂၅ တန်း အတိအကျ
    st.session_state['data_v82'] = st.data_editor(st.session_state['data_v82'], use_container_width=True, num_rows="fixed")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v82']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
