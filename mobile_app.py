import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v72", layout="wide")
SHEET_NAME = "LotteryData" 

def save_to_gsheet(data_to_save):
    try:
        if not data_to_save: 
            st.warning("သိမ်းစရာ ဒေတာ မရှိပါဘူးဗျ။")
            return False
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        if "gcp_service_account" not in st.secrets:
            st.error("GCP Secrets မတွေ့ပါ။ Streamlit Cloud Settings မှာ Key ထည့်ပေးပါ။")
            return False
            
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open(SHEET_NAME).get_worksheet(0)
        # ဂဏန်းရှေ့က 0 မပျောက်အောင် ' ခံသည်
        formatted_rows = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data_to_save]
        
        sheet.append_rows(formatted_rows, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_v72(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ပုံအောက်ပိုင်း ဝါးတာ/ကျဲတာကို နှိမ်ရန်
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(12,12))
    enhanced = clahe.apply(gray)
    
    results = reader.readtext(enhanced, paragraph=False, mag_ratio=2.0)
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 'text': t} for (b, t, p) in results]
    elements.sort(key=lambda k: k['y'])

    # 🎯 ROW MERGING (၂၅ တန်း ဝန်းကျင်ပဲ ဖြစ်စေရန်)
    rows = []
    if elements:
        curr_row = [elements[0]]
        for i in range(1, len(elements)):
            y_diff = elements[i]['y'] - curr_row[-1]['y']
            current_y = elements[i]['y']
            # အောက်ပိုင်းကျဲနေပါက ၇၅ pixel အထိ သည်းခံပြီး ပေါင်းပေးမည်
            dynamic_limit = 45 if current_y < (h/2) else 75
            
            if y_diff < dynamic_limit:
                # Column မထပ်အောင် စစ်ဆေးခြင်း
                is_overlap = any(abs(elements[i]['x'] - item['x']) < 100 for item in curr_row)
                if not is_overlap:
                    curr_row.append(elements[i])
                else:
                    rows.append(curr_row)
                    curr_row = [elements[i]]
            else:
                rows.append(curr_row)
                curr_row = [elements[i]]
        rows.append(curr_row)

    processed_data = []
    col_width = target_w / 4
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if c_idx > 3: c_idx = 3
            txt = item['text'].upper().strip()
            # Ditto Detection
            if re.search(r'[။၊"=“_…\.\-\']', txt) or (not txt.isdigit() and len(txt) == 1):
                row_cells[c_idx] = "DITTO"
            else:
                txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:]
                    else: row_cells[c_idx] = num
        
        if any(c != "" for c in row_cells):
            processed_data.append(row_cells)

    # Auto-Ditto Filling
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit(): last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v72 (Instant Scan)")
st.info("Scan ဖတ်ပြီးတာနဲ့ ဇယားကို ချက်ချင်း ပြသပေးပါမည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ဘယ်ဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="up_l")
with c2: up_right = st.file_uploader("ညာဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="up_r")

if st.button("🔍 Scan Now"):
    with st.spinner('ပုံကို စစ်ဆေးနေပါတယ် ခဏစောင့်ပါ...'):
        l_res, r_res = [], []
        if up_left: l_res = process_v72(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
        if up_right: r_res = process_v72(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
        max_r = max(len(l_res), len(r_res))
        final_table = []
        for i in range(max_r):
            l = l_res[i] if i < len(l_res) else ["","","",""]
            r = r_res[i] if i < len(r_res) else ["","","",""]
            final_table.append(l + r)
        
        st.session_state['data_v72'] = final_table
        st.rerun() # 🎯 ဇယားပေါ်လာအောင် Screen ကို Refresh လုပ်ခိုင်းခြင်း

# ဇယားကို ပြသခြင်း
if 'data_v72' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယားကွက်")
    # အသုံးပြုသူ ပြင်ဆင်နိုင်ရန် Data Editor
    st.session_state['data_v72'] = st.data_editor(st.session_state['data_v72'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        with st.spinner('Sheet ထဲ သိမ်းနေပါသည်...'):
            if save_to_gsheet(st.session_state['data_v72']):
                st.success("✅ Google Sheet ထဲ သိမ်းဆည်းပြီးပါပြီ!")
                st.balloons()
