import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v87", layout="wide")
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
    # Model ကို အသစ်ပြန်တင်ရန် cache clear လုပ်ထားပါသည်
    return easyocr.Reader(['en'], gpu=False)

def process_v87(uploaded_file):
    if uploaded_file is None: return [[""]*4]*25
    
    # 🎯 SAFE IMAGE LOADING (ပုံကို မှန်ကန်စွာ ဖတ်ရန်)
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    if img is None: return [[""]*4]*25

    reader = load_ocr()
    h, w = img.shape[:2]
    # ပုံကို AI ဖတ်ရလွယ်အောင် Standard Size ပြောင်းသည်
    img = cv2.resize(img, (1600, int(h * (1600 / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ပုံရိပ်ကို အလင်းအမှောင် ညှိနှိုင်းခြင်း (၃) မျိုး
    results = []
    results += reader.readtext(gray)
    results += reader.readtext(cv2.convertScaleAbs(gray, alpha=1.4, beta=10))
    
    # ၂၅ တန်း အတိအကျ Grid
    bins = np.linspace(40, img.shape[0]-40, 26) 
    grid = [["" for _ in range(4)] for _ in range(25)]
    
    col_w = 1600 // 4
    for (bbox, text, prob) in results:
        x_c = np.mean([p[0] for p in bbox])
        y_c = np.mean([p[1] for p in bbox])
        c_idx = int(x_c // col_w)
        if c_idx > 3: c_idx = 3
        
        # စာလုံးမှ ဂဏန်းသို့ ပြောင်းလဲခြင်း
        t = text.upper().replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0').replace('Z','2')
        
        if c_idx % 2 == 0: # ထိုးကွက် (၃ လုံး)
            val = re.sub(r'[^0-9]', '', t)
            if val: val = val.zfill(3)[-3:]
        else: # ထိုးကြေး (Amount)
            if re.search(r'[။၊"=“_…\.\-\']', t):
                val = "DITTO"
            else:
                val = re.sub(r'[^0-9]', '', t)

        b_idx = np.digitize(y_c, bins) - 1
        if 0 <= b_idx < 25:
            if grid[b_idx][c_idx] == "":
                grid[b_idx][c_idx] = val

    # Ditto Filling
    for c in [1, 3]:
        last = ""
        for r in range(25):
            if grid[r][c].isdigit() and grid[r][c] != "": last = grid[r][c]
            elif (grid[r][c] == "DITTO" or grid[r][c] == "") and last != "":
                grid[r][c] = last
                
    return grid

# --- UI ---
st.title("🔢 Lottery Pro v87 (Safe Loader)")
st.warning("ပုံတင်ပြီးနောက် 'Run Analysis' ခလုတ်ကို နှိပ်ပေးပါဗျ။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="up_l")
with c2: up_r = st.file_uploader("ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="up_r")

if st.button("🔍 Run Analysis"):
    if up_l or up_r:
        with st.spinner('AI မှ ဂဏန်းများကို ဖတ်နေပါသည်...'):
            l_res = process_v87(up_l) if up_l else [[""]*4]*25
            r_res = process_v87(up_r) if up_r else [[""]*4]*25
            
            # ဒေတာနှစ်ခုကို ပေါင်းစပ်ခြင်း
            st.session_state['final_data'] = [l_res[i] + r_res[i] for i in range(25)]
            st.rerun()
    else:
        st.error("ကျေးဇူးပြု၍ ပုံအရင်တင်ပေးပါဗျ။")

if 'final_data' in st.session_state:
    st.subheader("📝 စစ်ဆေးရန် ဇယား")
    # အကယ်၍ ဂဏန်းမပါလာသေးရင် ဒီဇယားထဲမှာတင် တိုက်ရိုက် ရိုက်ထည့်နိုင်ပါတယ်
    st.session_state['final_data'] = st.data_editor(
        st.session_state['final_data'], 
        use_container_width=True, 
        num_rows="fixed"
    )
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['final_data']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
