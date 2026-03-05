import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v77", layout="wide")
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

def clean_val(txt, is_num_col=True):
    """ဂဏန်းများကို သန့်စင်ရန်"""
    txt = txt.upper().replace(' ', '')
    mapping = {'S':'5', 'G':'6', 'B':'8', 'I':'1', 'O':'0', 'D':'0', 'Z':'2', 'A':'4', 'T':'7'}
    for k, v in mapping.items(): txt = txt.replace(k, v)
    
    if re.search(r'[။၊"=“_…\.\-\']', txt): return "DITTO"
    num = re.sub(r'[^0-9]', '', txt)
    if is_num_col and num: return num.zfill(3)[-3:]
    return num

def process_v77(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    img = cv2.resize(img, (1400, int(h * (1400 / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ၁။ ပုံတစ်ပြင်လုံးကို အရင်ဖတ် (Global Scan)
    global_results = reader.readtext(gray)
    
    # ၂။ တိုင်အလိုက် ခွဲဖတ် (Column Scan)
    col_data = [[] for _ in range(4)]
    w_step = 1400 // 4
    for i in range(4):
        crop = gray[:, i*w_step : (i+1)*w_step]
        res = reader.readtext(crop)
        for bbox, text, prob in res:
            y_c = np.mean([p[1] for p in bbox])
            val = clean_val(text, is_num_col=(i%2==0))
            if val: col_data[i].append((y_c, val))

    # 🎯 Hybrid Merge Logic: 
    # Global Scan ထဲက x position ကိုကြည့်ပြီး တိုင်တွေထဲ ပြန်ခွဲထည့်သည်
    for bbox, text, prob in global_results:
        x_c = np.mean([p[0] for p in bbox])
        y_c = np.mean([p[1] for p in bbox])
        c_idx = int(x_c // w_step)
        if c_idx < 4:
            val = clean_val(text, is_num_col=(c_idx%2==0))
            if val: col_data[c_idx].append((y_c, val))

    # Y-coordinate အလိုက် စီပြီး ၂၅ တန်းဝန်းကျင် ဖြစ်အောင် ညှိသည်
    bins = np.linspace(0, img.shape[0], 27)
    final_rows = [["" for _ in range(4)] for _ in range(len(bins)-1)]
    
    for c_idx in range(4):
        # ထပ်နေတဲ့ data တွေကို ဖယ်ပြီး y အလိုက် စီသည်
        sorted_items = sorted(col_data[c_idx], key=lambda x: x[0])
        for y, v in sorted_items:
            b_idx = np.digitize(y, bins) - 1
            if 0 <= b_idx < len(final_rows):
                # အကယ်၍ အကွက်ထဲမှာ ရှိနှင့်နေပြီးသားဆိုရင် မထည့်တော့ပါ (Duplicate ရှောင်ရန်)
                if final_rows[b_idx][c_idx] == "":
                    final_rows[b_idx][c_idx] = v

    cleaned = [r for r in final_rows if any(c != "" for c in r)]
    
    # Auto-Fill Ditto
    for c in [1, 3]:
        last = ""
        for r in range(len(cleaned)):
            if cleaned[r][c].isdigit(): last = cleaned[r][c]
            elif (cleaned[r][c] == "DITTO" or cleaned[r][c] == "") and last != "":
                cleaned[r][c] = last
    return cleaned

# --- UI ---
st.title("🔢 Lottery Pro v77 (The Hybrid Model)")
st.info("ယခင် Version များ၏ အားသာချက်များကို ပေါင်းစပ်ထားသော Final Version ဖြစ်ပါသည်။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_r = st.file_uploader("ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 Intelligent Scan"):
    with st.spinner('နည်းလမ်းနှစ်မျိုးဖြင့် ပေါင်းစပ်စစ်ဆေးနေပါသည်...'):
        l_res, r_res = [], []
        if up_l: l_res = process_v77(cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), 1))
        if up_r: r_res = process_v77(cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), 1))
            
        max_r = max(len(l_res), len(r_res))
        final = []
        for i in range(max_r):
            l = l_res[i] if i < len(l_res) else ["","","",""]
            r = r_res[i] if i < len(r_res) else ["","","",""]
            final.append(l + r)
        
        st.session_state['data_v77'] = final
        st.rerun()

if 'data_v77' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    st.session_state['data_v77'] = st.data_editor(st.session_state['data_v77'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v77']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
