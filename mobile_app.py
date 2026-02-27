import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gc
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- BASIC CONFIG ---
st.set_page_config(page_title="Lottery Scanner v31", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM ချွေတာရန် GPU ပိတ်ထားပြီး အပေါ့ပါးဆုံး Model သုံးမည်
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v31(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        # '0' ရှေ့ကမပျောက်စေရန် Formatting လုပ်ခြင်း
        formatted = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Mobile 8-Column Scanner v31")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.info("V31: ၈ တိုင်ဗောက်ချာများကို ဖုန်းတွင် Memory Crash လုံးဝမဖြစ်စေရန် အထူးပြုပြင်ထားသည်။")

up_file = st.file_uploader("၈ တိုင်ဗောက်ချာပုံကို ရွေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v31(img, n_cols):
    reader = load_ocr()
    h, w = img.shape[:2]
    
    # ဖုန်း RAM နှင့်ကိုက်ညီအောင် Resolution ကို ညှိခြင်း
    target_w = 1100
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # စာလုံးများ ပိုမိုထင်ရှားစေရန် Contrast မြှင့်ခြင်း
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    h_gray = gray.shape[0]
    all_results = []
    
    # ပုံကို ၈ ပိုင်းအထိ ခွဲဖတ်ခြင်းဖြင့် RAM ဝန်ကို အနည်းဆုံးဖြစ်စေသည်
    num_parts = 8
    progress_bar = st.progress(0)
    
    for i in range(num_parts):
        y1 = max(0, int(h_gray * (i/num_parts)) - 30)
        y2 = min(h_gray, int(h_gray * ((i+1)/num_parts)) + 30)
        segment = gray[y1:y2, :]
        
        res = reader.readtext(segment, paragraph=False, width_ths=0.4)
        for (bbox, text, prob) in res:
            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox]) + y1
            all_results.append({'x': cx, 'y': cy, 'text': text})
        
        progress_bar.progress((i + 1) / num_parts)
        del segment
        gc.collect()

    if not all_results: return []

    # ROW CLUSTERING
    all_results.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 22
    current_row = [all_results[0]]
    for i in range(1, len(all_results)):
        if all_results[i]['y'] - current_row[-1]['y'] < y_threshold:
            current_row.append(all_results[i])
        else:
            rows_list.append(current_row)
            current_row = [all_results[i]]
    rows_list.append(current_row)

    col_edges = np.linspace(0, target_w, n_cols + 1)
    final_grid = []

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        bins = [[] for _ in range(n_cols)]
        for item in row_items:
            c_idx = int(np.searchsorted(col_edges, item['x']) - 1)
            if 0 <= c_idx < n_cols: bins[c_idx].append(item)
        
        for c in range(n_cols):
            bins[c].sort(key=lambda k: k['x'])
            txt = "".join([i['text'].upper() for i in bins[c]]).strip()
            
            # --- ADVANCED DITTO & NOISE FIX ---
            # လက်ရေး "။" ကို AI မှ မှားဖတ်လေ့ရှိသော စာလုံးပုံစံများ
            is_ditto = any(m in txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '/', '11', 'I', '(', ')', 'N', 'H'])
            
            if is_ditto and len(re.sub(r'[^0-9]', '', txt)) < 3:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c % 2 == 0: row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: row_cells[c] = num
        final_grid.append(row_cells)

    # Smart Fill Down (Ditto အတွက် အပေါ်တန်ဖိုးယူခြင်း)
    for c in range(n_cols):
        if c % 2 != 0: 
            last_amt = ""
            for r in range(len(final_grid)):
                val = str(final_grid[r][c]).strip()
                if (val == "DITTO" or val == "") and last_amt:
                    final_grid[r][c] = last_amt
                elif val != "DITTO" and val != "":
                    last_amt = val
        else:
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400, caption="Uploaded Image")
    
    if st.button(f"🔍 ၈ တိုင်စလုံး Scan ဖတ်မည်"):
        with st.spinner("ဖုန်း Memory ချွေတာပြီး အသေးစိတ် ဖတ်နေပါသည်..."):
            res = process_v31(img, a_cols)
            st.session_state['data_v31'] = res

if 'data_v31' in st.session_state:
    edited = st.data_editor(st.session_state['data_v31'], use_container_width=True)
    if st.button("💾 Google Sheet ထဲသို့ ပို့မည်"):
        if save_to_sheets_v31(edited):
            st.success("အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
