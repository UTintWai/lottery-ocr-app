import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v29", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM ချွေတာရန်နှင့် ဖုန်းတွင် Crash မဖြစ်ရန်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def save_to_sheets_v29(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        # '062' ကဲ့သို့ ပေါ်ရန် formatting ထည့်မည်
        formatted = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 High-Precision Lottery Scanner v29")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.info("V29: ဖုန်းတွင် Memory Crash ဖြစ်ခြင်းကို ကာကွယ်ရန်နှင့် လက်ရေး Ditto ကို ပိုမိုမှန်ကန်စွာ ဖတ်ရန် ပြင်ဆင်ထားသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v29(img, n_cols):
    h, w = img.shape[:2]
    # Resolution ကို RAM နှင့် ကိုက်ညီအောင် ညှိထားသည်
    target_w = 1100
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ပုံကို ၄ ပိုင်းခွဲဖတ်ခြင်းဖြင့် ဖုန်းတွင် RAM Crash မဖြစ်အောင် လုပ်ဆောင်သည်
    h_gray = gray.shape[0]
    results = []
    num_parts = 4
    for i in range(num_parts):
        y1 = max(0, int(h_gray * (i/num_parts)) - 30)
        y2 = min(h_gray, int(h_gray * ((i+1)/num_parts)) + 30)
        seg = gray[y1:y2, :]
        res = reader.readtext(seg, paragraph=False, link_threshold=0.3)
        for (bbox, text, prob) in res:
            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox]) + y1
            results.append({'x': cx, 'y': cy, 'text': text})

    if not results: return []

    # ROW CLUSTERING
    results.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 22
    current_row = [results[0]]
    for i in range(1, len(results)):
        if results[i]['y'] - current_row[-1]['y'] < y_threshold:
            current_row.append(results[i])
        else:
            rows_list.append(current_row)
            current_row = [results[i]]
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
            
            # --- IMPROVED DITTO LOGIC FOR HANDWRITING ---
            # လက်ရေး "။" ကို AI မှ 4, u, v, n, 11, /, i, ( ) စသည်ဖြင့် မှားဖတ်သည်ကို Ditto ဟု သတ်မှတ်မည်
            is_ditto_pattern = any(m in txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '/', '11', 'I', '(', ')', 'N'])
            
            if is_ditto_pattern and len(re.sub(r'[^0-9]', '', txt)) < 3:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c % 2 == 0: row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: row_cells[c] = num
        final_grid.append(row_cells)

    # Smart Fill Down Logic
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
    st.image(img, width=400)
    
    if st.button(f"🔍 Scan {a_cols} Columns"):
        with st.spinner("ဖုန်းအတွက် Memory ချွေတာပြီး အမှားအယွင်းမရှိအောင် ဖတ်နေပါသည်..."):
            res = process_v29(img, a_cols)
            st.session_state['data_v29'] = res

if 'data_v29' in st.session_state:
    edited = st.data_editor(st.session_state['data_v29'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets_v29(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
