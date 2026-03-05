import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v78", layout="wide")
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

def clean_and_validate(txt, is_num_col=True):
    """ဂဏန်းများကို ပြုပြင်သန့်စင်ခြင်း"""
    txt = txt.upper().strip()
    # လက်ရေးစာများတွင် မှားတတ်သော Mapping
    mapping = {'S':'5', 'G':'6', 'B':'8', 'I':'1', 'L':'1', 'O':'0', 'D':'0', 'Z':'2', 'A':'4', 'T':'7'}
    for k, v in mapping.items():
        txt = txt.replace(k, v)
    
    if re.search(r'[။၊"=“_…\.\-\']', txt):
        return "DITTO"
    
    num = re.sub(r'[^0-9]', '', txt)
    if is_num_col and num:
        return num.zfill(3)[-3:]
    return num

def process_v78(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resize to standardized size
    img = cv2.resize(img, (1500, int(h * (1500 / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ၁။ Global Scan (Confidence စစ်ထုတ်ခြင်း)
    results = reader.readtext(gray)
    
    # ၂။ Row-based Grouping with Strict Y-Coordinates
    row_data = []
    for (bbox, text, prob) in results:
        y_c = np.mean([p[1] for p in bbox])
        x_c = np.mean([p[0] for p in bbox])
        c_idx = int(x_c // (1500/4))
        if c_idx > 3: c_idx = 3
        
        val = clean_and_validate(text, is_num_col=(c_idx%2==0))
        if val:
            row_data.append({'y': y_c, 'c': c_idx, 'v': val})
            
    # Y-coordinate အလိုက် Binning (၂၅ တန်းအထိ ကန့်သတ်ရန်)
    bins = np.linspace(0, img.shape[0], 27)
    grid = [["" for _ in range(4)] for _ in range(len(bins)-1)]
    
    for item in row_data:
        b_idx = np.digitize(item['y'], bins) - 1
        if 0 <= b_idx < len(grid):
            # အကယ်၍ data ရှိနှင့်ပြီးသားဆိုလျှင် မထည့်ဘဲ ရှောင်သည်
            if grid[b_idx][item['c']] == "":
                grid[b_idx][item['c']] = item['v']

    # အလွတ်များဖယ်ထုတ်ခြင်း
    final_data = [r for r in grid if any(c != "" for c in r)]
    
    # Smart Ditto Fill
    for c in [1, 3]:
        last = ""
        for r in range(len(final_data)):
            if final_data[r][c].isdigit(): last = final_data[r][c]
            elif (final_data[r][c] == "DITTO" or final_data[r][c] == "") and last != "":
                final_data[r][c] = last
                
    return final_data

# --- UI ---
st.title("🔢 Lottery Pro v78 (Consensus Mode)")
st.markdown("ယခင် Version များတွင် မှန်ကန်ခဲ့သော အားသာချက်များကို ပြန်လည်စုစည်းထားပါသည်။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_r = st.file_uploader("ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 High-Speed Scan"):
    with st.spinner('ပုံရိပ်များကို အတည်ပြုနေပါသည်...'):
        l_res, r_res = [], []
        if up_l: l_res = process_v78(cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), 1))
        if up_r: r_res = process_v78(cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), 1))
            
        max_r = max(len(l_res), len(r_res))
        final = []
        for i in range(max_r):
            l = l_res[i] if i < len(l_res) else ["","","",""]
            r = r_res[i] if i < len(r_res) else ["","","",""]
            final.append(l + r)
        
        st.session_state['data_v78'] = final
        st.rerun()

if 'data_v78' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    # Data Editor ကို Session State နှင့် တိုက်ရိုက်ချိတ်ဆက်ခြင်း
    st.session_state['data_v78'] = st.data_editor(st.session_state['data_v78'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v78']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
