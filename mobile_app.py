import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v79", layout="wide")
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

def process_v79(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # ပုံကို အကြီးဆုံးချဲ့ဖတ်မည်
    img = cv2.resize(img, (1800, int(h * (1800 / w))))
    
    # ရိုးရှင်းသော Grayscale သာသုံးသည်
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray, paragraph=False, detail=1)
    
    # Y-coordinate အလိုက် Binning (၂၅ တန်း အတိအကျရရန်)
    bins = np.linspace(0, img.shape[0], 26) # ၂၅ ကွက် တိတိ
    grid = [["" for _ in range(4)] for _ in range(25)]
    
    col_w = 1800 // 4
    for (bbox, text, prob) in results:
        x_c = np.mean([p[0] for p in bbox])
        y_c = np.mean([p[1] for p in bbox])
        
        c_idx = int(x_c // col_w)
        if c_idx > 3: c_idx = 3
        
        # အခြေခံ ဂဏန်းသန့်စင်မှု
        t = text.upper().replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
        if re.search(r'[။၊"=“_…\.\-\']', t):
            val = "DITTO"
        else:
            val = re.sub(r'[^0-9]', '', t)
            if (c_idx % 2 == 0) and val: val = val.zfill(3)[-3:]

        b_idx = np.digitize(y_c, bins) - 1
        if 0 <= b_idx < 25:
            if grid[b_idx][c_idx] == "": grid[b_idx][c_idx] = val

    # Smart Ditto Fill
    for c in [1, 3]:
        last = ""
        for r in range(25):
            if grid[r][c].isdigit(): last = grid[r][c]
            elif (grid[r][c] == "DITTO" or grid[r][c] == "") and last != "":
                grid[r][c] = last
    return grid

# --- UI ---
st.title("🔢 Lottery Pro v79 (Manual Check Focus)")
st.warning("AI မှားနိုင်ခြေရှိပါသည်။ ကျေးဇူးပြု၍ ဇယားတွင် ဂဏန်းများကို အမြန်စစ်ဆေးပေးပါဗျ။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_r = st.file_uploader("ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 Scan Everything"):
    with st.spinner('ဖတ်နေပါသည်...'):
        l_res = process_v79(cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), 1)) if up_l else [[""]*4]*25
        r_res = process_v79(cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), 1)) if up_r else [[""]*4]*25
        
        final = [l_res[i] + r_res[i] for i in range(25)]
        st.session_state['data_v79'] = final
        st.rerun()

if 'data_v79' in st.session_state:
    st.subheader("📝 စစ်ဆေးရန် (အမှားတွေ့ပါက တန်းပြင်ပါ)")
    # ၂၅ တန်း အတိအကျ ပြပေးထားပါသည်
    st.session_state['data_v79'] = st.data_editor(
        st.session_state['data_v79'], 
        use_container_width=True, 
        num_rows="fixed", # အတန်းအရေအတွက်ကို ကန့်သတ်ထားသည်
        column_config={i: st.column_config.TextColumn(width="medium") for i in range(8)}
    )
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v79']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
