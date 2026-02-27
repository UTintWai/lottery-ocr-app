import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gc
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v30", layout="wide")

@st.cache_resource
def load_ocr():
    # ဖုန်း RAM အတွက် အပေါ့ပါးဆုံး model configuration
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='./model', download_enabled=True)

def save_to_sheets_v30(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        formatted = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 8-Column Mobile Scanner v30")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.warning("V30: ၈ တိုင်ဗောက်ချာများကို ဖုန်းဖြင့်ဖတ်လျှင် Memory မပြည့်စေရန် အပိုင်း ၆ ပိုင်းခွဲ၍ ဖတ်ပေးမည်။")

up_file = st.file_uploader("၈ တိုင်ဗောက်ချာပုံကို ရွေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v30(img, n_cols):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1200
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # --- ULTRA MEMORY MANAGEMENT ---
    # ပုံကို အပိုင်း ၆ ပိုင်းအထိ ခွဲလိုက်ခြင်းဖြင့် တစ်ကြိမ်လျှင် RAM အနည်းငယ်သာ သုံးမည်
    h_gray = gray.shape[0]
    all_results = []
    num_parts = 6
    
    for i in range(num_parts):
        y1 = max(0, int(h_gray * (i/num_parts)) - 40)
        y2 = min(h_gray, int(h_gray * ((i+1)/num_parts)) + 40)
        segment = gray[y1:y2, :]
        
        # အပိုင်းလိုက်ဖတ်ခြင်း
        res = reader.readtext(segment, paragraph=False, width_ths=0.5, add_margin=0.1)
        for (bbox, text, prob) in res:
            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox]) + y1
            all_results.append({'x': cx, 'y': cy, 'text': text})
        
        # RAM ကို ပြန်ရှင်းထုတ်ခြင်း
        del segment
        gc.collect()

    if not all_results: return []

    # ROW CLUSTERING (စာကြောင်းအကွာအဝေး ညှိခြင်း)
    all_results.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 20
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
            
            # လက်ရေး Ditto (။) နှင့် အလားတူအမှားများကို စစ်ဆေးခြင်း
            ditto_patterns = ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '/', '11', 'I', '(', ')', 'N', 'W', 'M', 'H']
            is_ditto = any(m in txt for m in ditto_patterns)
            
            if is_ditto and len(re.sub(r'[^0-9]', '', txt)) < 3:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c % 2 == 0: row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: row_cells[c] = num
        final_grid.append(row_cells)

    # Smart Fill Down
    for c in range(n_cols):
        if c % 2 != 0: 
            last_val = ""
            for r in range(len(final_grid)):
                v = str(final_grid[r][c])
                if (v == "DITTO" or v == "") and last_val: final_grid[r][c] = last_val
                elif v != "DITTO" and v != "": last_val = v
        else:
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button(f"🔍 Scan All {a_cols} Columns"):
        with st.spinner("Memory ချွေတာပြီး ၈ တိုင်စလုံးကို အသေးစိတ် ဖတ်နေပါသည်..."):
            res = process_v30(img, a_cols)
            st.session_state['data_v30'] = res

if 'data_v30' in st.session_state:
    edited = st.data_editor(st.session_state['data_v30'], use_container_width=True)
    if st.button("💾 Google Sheet ထဲသိမ်းမည်"):
        if save_to_sheets_v30(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
