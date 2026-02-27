import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gc
import gspread
import time
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Scanner v32", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v32(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        # --- STABLE SAVING LOGIC ---
        # Connection မပြတ်အောင် တစ်ကြောင်းချင်းစီ ပို့မည်
        progress_text = st.empty()
        count = 0
        for row in data:
            formatted_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row]
            if any(formatted_row): # အလွတ်မဟုတ်မှ ပို့မည်
                sheet.append_row(formatted_row)
                count += 1
                progress_text.text(f"သိမ်းဆည်းနေဆဲ... ({count}/{len(data)} ကြောင်း)")
                time.sleep(0.1) # Google API limit မထိအောင် ခဏနားသည်
        return True
    except Exception as e:
        st.error(f"Sheet ထဲ ဒေတာမဝင်ရသည့်အကြောင်းရင်း: {str(e)}")
        return False

st.title("🔢 Data Saving Pro v32")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.success("V32: Sheet ထဲမဝင်သည့်ပြဿနာကို ဖြေရှင်းထားသည်။")

up_file = st.file_uploader("ပုံတင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v32(img, n_cols):
    reader = load_ocr()
    h, w = img.shape[:2]
    # RAM ပိုချွေတာရန် width ကို လျှော့ချသည်
    target_w = 1000 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ၈ တိုင်စလုံး မိအောင် အပိုင်း ၈ ပိုင်းခွဲဖတ်မည်
    h_gray = gray.shape[0]
    all_results = []
    num_parts = 8
    
    for i in range(num_parts):
        y1 = max(0, int(h_gray * (i/num_parts)) - 25)
        y2 = min(h_gray, int(h_gray * ((i+1)/num_parts)) + 25)
        segment = gray[y1:y2, :]
        res = reader.readtext(segment, paragraph=False, detail=1)
        for (bbox, text, prob) in res:
            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox]) + y1
            all_results.append({'x': cx, 'y': cy, 'text': text})
        del segment
        gc.collect()

    if not all_results: return []

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
        for item in row_items:
            c_idx = int(np.searchsorted(col_edges, item['x']) - 1)
            if 0 <= c_idx < n_cols:
                # ဂဏန်းမဟုတ်သော စာသားများပါက ဖျက်ထုတ်မည်
                clean_txt = re.sub(r'[^0-9"။=LL/UVYI\(\)]', '', item['text'].upper())
                row_cells[c_idx] = clean_txt
        final_grid.append(row_cells)

    # Smart Formatting & Fill Down
    for r in range(len(final_grid)):
        for c in range(n_cols):
            val = final_grid[r][c]
            # ဂဏန်း ၃ လုံးဖြစ်အောင် ညှိခြင်း
            if c % 2 == 0 and val.isdigit():
                final_grid[r][c] = val.zfill(3)
            # Ditto Logic
            elif any(m in val for m in ['"', '။', '=', 'L', '/', 'V', 'U', 'Y', 'I']):
                final_grid[r][c] = "DITTO"

    # Fill Amount
    for c in range(1, n_cols, 2):
        last_v = ""
        for r in range(len(final_grid)):
            if final_grid[r][c] == "DITTO" and last_v: final_grid[r][c] = last_v
            elif final_grid[r][c].isdigit(): last_v = final_grid[r][c]
    
    # Clean Numbers
    for c in range(0, n_cols, 2):
        for r in range(len(final_grid)):
            if final_grid[r][c] == "DITTO": final_grid[r][c] = ""

    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button(f"🔍 {a_cols} တိုင် Scan ဖတ်မည်"):
        with st.spinner("ဖတ်နေပါသည်..."):
            res = process_v32(img, a_cols)
            st.session_state['data_v32'] = res

if 'data_v32' in st.session_state:
    edited = st.data_editor(st.session_state['data_v32'], use_container_width=True)
    if st.button("💾 Google Sheet ထဲသို့ အသေအချာ သိမ်းမည်"):
        if save_to_sheets_v32(edited):
            st.success("Sheet ထဲသို့ ဒေတာများ အကုန်ရောက်သွားပါပြီ!")
