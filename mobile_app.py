import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v68", layout="wide")
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

def process_v68(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ပုံကို ပိုကြည်လင်အောင် CLAHE နဲ့ Denoise လုပ်သည်
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    dst = cv2.medianBlur(gray, 3)

    results = reader.readtext(dst, paragraph=False, mag_ratio=1.5)
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 
                 'w': (b[1][0] - b[0][0]), 'text': t} for (b, t, p) in results]
    
    # ၁။ ROW CLUSTERING (စာကြောင်းများကို AI နည်းဖြင့် အကွာအဝေးတွက်ခွဲခြင်း)
    elements.sort(key=lambda k: k['y'])
    rows = []
    if elements:
        curr_row = [elements[0]]
        # Threshold ကို ၃၅ မှာ ပုံသေထားပြီး ဂဏန်းထပ်တာကို ရှောင်သည်
        for i in range(1, len(elements)):
            if elements[i]['y'] - curr_row[-1]['y'] < 35:
                # ဂဏန်းတွေ တစ်ခုနဲ့တစ်ခု အရမ်းနီးကပ်နေရင် (ထပ်နေရင်) မပေါင်းပါနဲ့
                is_overlap = any(abs(elements[i]['x'] - item['x']) < 50 for item in curr_row)
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
    # ၂။ COLUMN MAPPING (X-coordinate ကို ၄ ပိုင်း အချိုးကျ ခွဲခြင်း)
    col_width = target_w / 4
    
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            x_pos = item['x']
            c_idx = int(x_pos // col_width)
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
        
        if any(row_cells):
            processed_data.append(row_cells)

    # ၃။ SMART DITTO FILL (အပေါ်ကတန်ဖိုးကို ဆွဲချခြင်း)
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit(): last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v68 (Anti-Overlap)")
st.info("ဂဏန်းများ ထပ်သွားခြင်း (Merge ဖြစ်ခြင်း) ကို ပြင်ဆင်ထားသော Version ဖြစ်ပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ပုံ (၁) - ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'])
with c2: up_right = st.file_uploader("ပုံ (၂) - ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'])

if st.button("🔍 Scan & Fix Rows"):
    l_res, r_res = [], []
    if up_left: l_res = process_v68(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
    if up_right: r_res = process_v68(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
    max_r = max(len(l_res), len(r_res))
    final = []
    for i in range(max_r):
        l = l_res[i] if i < len(l_res) else ["","","",""]
        r = r_res[i] if i < len(r_res) else ["","","",""]
        final.append(l + r)
    st.session_state['v68_data'] = final

if 'v68_data' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    edited = st.data_editor(st.session_state['v68_data'], use_container_width=True, num_rows="dynamic")
    
    col_s, col_c = st.columns(2)
    with col_s:
        if st.button("💾 Save to Google Sheet"):
            if save_to_gsheet(edited): st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
    with col_c:
        df = pd.DataFrame(edited, columns=['A','B','C','D','E','F','G','H'])
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "lottery.csv")
