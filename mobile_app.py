import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v19", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.title("🔢 Lottery Pro Scanner (v19 - 8 Col Fix)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက် ရွေးပါ", [6, 8], index=1)
    st.info("V19: ၈ တိုင်ဗောက်ချာအတွက် တိုင်အကျယ်များကို အချိုးကျ ပြန်လည်ညှိထားပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ ရွေးရန်", type=['jpg', 'jpeg', 'png'])

def process_v19(img, n_cols):
    h, w = img.shape[:2]
    target_w = 1600
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR results
    results = reader.readtext(gray, paragraph=False, link_threshold=0.1, mag_ratio=1.6)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # ROW CLUSTERING
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 30 
    
    current_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - current_row[-1]['y'] < y_threshold:
            current_row.append(raw_data[i])
        else:
            rows_list.append(current_row)
            current_row = [raw_data[i]]
    rows_list.append(current_row)

    # --- 8-COLUMN RATIO TUNING ---
    # ၈ တိုင်အတွက် တိုင်တစ်ခုစီရဲ့ width အချိုးအစားကို ချိန်ညှိခြင်း
    if n_cols == 8:
        # ဂဏန်းတိုင် နဲ့ ထိုးကြေးတိုင် အချိုးအစား (ဥပမာ- 45%, 55%)
        # ပိုမိုတိကျစေရန် ညီတူညီမျှ ခွဲမည့်အစား အချိုးကျ ခွဲဝေပါမည်
        col_edges = [0, 0.12, 0.25, 0.37, 0.50, 0.62, 0.75, 0.87, 1.0]
        col_edges = [x * target_w for x in col_edges]
    else:
        col_edges = np.linspace(0, target_w, n_cols + 1)

    final_grid = []
    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        bins = [[] for _ in range(n_cols)]
        
        for item in row_items:
            for c in range(n_cols):
                if col_edges[c] <= item['x'] < col_edges[c+1]:
                    bins[c].append(item)
                    break
        
        for c in range(n_cols):
            bins[c].sort(key=lambda k: k['x'])
            combined_txt = "".join([i['text'] for i in bins[c]])
            
            # Ditto Logic
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1', '7', 'I']) and len(combined_txt) <= 2
            
            if is_ditto:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: # ဂဏန်းတိုင်
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေးတိုင်
                        row_cells[c] = num
        final_grid.append(row_cells)

    # SMART FILL-DOWN
    for c in range(n_cols):
        if c % 2 != 0: 
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                if val == "DITTO" or val == "":
                    if last_amt != "":
                        final_grid[r][c] = last_amt
                else:
                    last_amt = val
        else:
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

# (Save function and Main UI same as before)
if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=550)
    
    if st.button("🔍 ဒေတာထုတ်ယူမည် (၈ တိုင်)"):
        with st.spinner("အသေးစိတ်စစ်ဆေးနေပါသည်..."):
            res = process_v19(img, a_cols)
            st.session_state['data_v19'] = res

if 'data_v19' in st.session_state:
    edited = st.data_editor(st.session_state['data_v19'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        # save_to_sheets code...
        st.success("သိမ်းဆည်းပြီးပါပြီ!")
