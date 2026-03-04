import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v75", layout="wide")
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

def process_v75(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resolution ကို သင့်တင့်ရုံပဲ ထားသည် (စာလုံးတွေ မကွဲစေရန်)
    target_w = 1200 
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # 🎯 ရိုးရိုး Grayscale ပဲသုံးသည် (Filter အပိုတွေ မပါ)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    results = reader.readtext(gray, paragraph=False)
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 'text': t} for (b, t, p) in results]
    elements.sort(key=lambda k: k['y'])

    # 🎯 ROW STABILIZER: ၂၅ တန်းဝန်းကျင်ပဲ ထွက်လာအောင် Threshold ကို ညှိခြင်း
    rows = []
    if elements:
        curr_row = [elements[0]]
        for i in range(1, len(elements)):
            # အကွာအဝေး ၅၀ Pixel အတွင်းရှိရင် တစ်ကြောင်းတည်းဟု သတ်မှတ်သည်
            if elements[i]['y'] - curr_row[-1]['y'] < 50:
                curr_row.append(elements[i])
            else:
                rows.append(curr_row)
                curr_row = [elements[i]]
        rows.append(curr_row)

    processed_data = []
    col_split = target_w / 4
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_split)
            if c_idx > 3: c_idx = 3
            
            txt = item['text'].upper().replace(' ', '')
            # Ditto Detection
            if re.search(r'[။၊"=“_…\.\-\']', txt):
                row_cells[c_idx] = "DITTO"
            else:
                # Basic Error Fixes
                txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:]
                    else: row_cells[c_idx] = num
        
        if any(row_cells): processed_data.append(row_cells)

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
st.title("🔢 Lottery Pro v75 (Stabilized Mode)")
st.info("Filter အပိုများကို ဖြုတ်ထားပြီး အတန်း ၂၅ တန်းရရှိရေးကို ဦးစားပေးထားပါသည်။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ်ဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_r = st.file_uploader("ညာဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 Run Scan"):
    l_res, r_res = [], []
    if up_l: l_res = process_v75(cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), 1))
    if up_r: r_res = process_v75(cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), 1))
            
    max_r = max(len(l_res), len(r_res))
    final = []
    for i in range(max_r):
        l = l_res[i] if i < len(l_res) else ["","","",""]
        r = r_res[i] if i < len(r_res) else ["","","",""]
        final.append(l + r)
    
    st.session_state['data_v75'] = final
    st.rerun()

if 'data_v75' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    st.session_state['data_v75'] = st.data_editor(st.session_state['data_v75'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v75']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!")
