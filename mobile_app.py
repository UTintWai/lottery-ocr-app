import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v20", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.title("🔢 Multi-Column Lottery Scanner (v20)")

with st.sidebar:
    # အသုံးပြုသူ ရိုက်ထားသော တိုင်အရေအတွက်ကို ရွေးရန်
    a_cols = st.selectbox("ဗောက်ချာပါ တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.info(f"ယခု {a_cols} တိုင်ဗောက်ချာအတွက် Logic ကို အသုံးပြုနေပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ ရွေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v20(img, n_cols):
    h, w = img.shape[:2]
    # Resolution ကို ပိုကောင်းအောင် 1800px ထားပါမယ်
    target_w = 1800
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR results: ၈ တိုင်အတွက် link_threshold ကို အထူးလျှော့ထားသည်
    results = reader.readtext(gray, paragraph=False, link_threshold=0.05, mag_ratio=1.6)
    
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

    # DYNAMIC GRID CALCULATION
    # တိုင်များ များလာလျှင် boundary များကို ပိုမိုစိပ်အောင် တွက်ချက်ခြင်း
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
            
            # Ditto Logic (။ သင်္ကေတနှင့် တူသည်များကို အားလုံးဖမ်းမည်)
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1', '7', 'I', '/', '(', ')']) and len(combined_txt) <= 2
            
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

    # AUTO-FILL LOGIC
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်အတွက်သာ
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

# --- UI LOGIC ---
if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=600)
    
    if st.button(f"🔍 {a_cols} တိုင် Scan လုပ်မည်"):
        with st.spinner("AI စနစ်ဖြင့် အကွက်များကို ခွဲခြားနေပါသည်..."):
            res = process_v20(img, a_cols)
            st.session_state['data_v20'] = res

if 'data_v20' in st.session_state:
    st.success("ဖတ်ခြင်း ပြီးဆုံးပါပြီ။ အောက်ပါဇယားတွင် တိုက်ဆိုင်စစ်ဆေးပါ။")
    edited = st.data_editor(st.session_state['data_v20'], use_container_width=True)
    # Google Sheet save logic...
