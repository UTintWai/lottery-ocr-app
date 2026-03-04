import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v67", layout="wide")
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

def process_v67(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Noise တွေကို ဖယ်ပြီး စာလုံးကို ပိုထင်ရှားစေခြင်း
    dst = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    results = reader.readtext(dst, paragraph=False, mag_ratio=2.0)
    
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 'text': t} for (b, t, p) in results]
    elements.sort(key=lambda k: k['y'])

    # 🎯 ROW MERGING LOGIC: စာကြောင်းအပိုတွေ မထွက်အောင် ၄၅ pixel အတွင်းကို တစ်ကြောင်းတည်းပေါင်းမည်
    rows = []
    if elements:
        curr_row = [elements[0]]
        for i in range(1, len(elements)):
            # ဒီနေရာမှာ Threshold ကို ၄၅ အထိ မြှင့်လိုက်လို့ ၂၅ တန်းပဲ ထွက်လာဖို့ ကူညီပါလိမ့်မယ်
            if elements[i]['y'] - curr_row[-1]['y'] < 45: 
                curr_row.append(elements[i])
            else:
                rows.append(curr_row)
                curr_row = [elements[i]]
        rows.append(curr_row)

    processed_data = []
    for r_items in rows:
        r_items.sort(key=lambda k: k['x'])
        row_cells = ["" for _ in range(4)]
        
        # တကယ့်ဂဏန်းပါတဲ့ စာသားတွေကိုပဲ ရွေးထုတ်သည်
        for item in r_items:
            txt = item['text'].upper().strip()
            x = item['x']
            # တိုင်ခွဲခြားမှုကို x position အပေါ်မူတည်ပြီး ပိုတိကျအောင်လုပ်သည်
            col_idx = int(x // (target_w / 4))
            if col_idx > 3: col_idx = 3
            
            if re.search(r'[။၊"=“_…\.\-\']', txt):
                row_cells[col_idx] = "DITTO"
            else:
                txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if col_idx % 2 == 0: row_cells[col_idx] = num.zfill(3)[-3:]
                    else: row_cells[col_idx] = num
        
        # အနည်းဆုံး ဂဏန်းတစ်ခုပါမှ Row ထဲထည့်မည် (စာကြောင်းအပိုတွေ ဖယ်ထုတ်ရန်)
        if any(row_cells):
            processed_data.append(row_cells)

    # Auto-Fill Ditto
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit(): last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v67 (Row Stabilizer)")
st.info("စာကြောင်းရေ အပိုထွက်ခြင်း (၃၀ တန်းဖြစ်နေခြင်း) ကို ပြင်ဆင်ထားသည့် Version ဖြစ်ပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ပုံ (၁) - ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'])
with c2: up_right = st.file_uploader("ပုံ (၂) - ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'])

if st.button("🔍 Scan and Merge Rows"):
    l_data, r_data = [], []
    if up_left: l_data = process_v67(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
    if up_right: r_data = process_v67(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
    max_r = max(len(l_data), len(r_data))
    final = []
    for i in range(max_r):
        l_row = l_data[i] if i < len(l_data) else ["","","",""]
        r_row = r_data[i] if i < len(r_data) else ["","","",""]
        final.append(l_row + r_row)
    st.session_state['v67_data'] = final

if 'v67_data' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (၂၅ တန်း ဝန်းကျင်)")
    edited = st.data_editor(st.session_state['v67_data'], use_container_width=True, num_rows="dynamic")
    
    c_save, c_csv = st.columns(2)
    with c_save:
        if st.button("💾 Save to Google Sheet"):
            if save_to_gsheet(edited):
                st.success("✅ သိမ်းဆည်းပြီးပါပြီ!")
                st.balloons()
    with c_csv:
        df = pd.DataFrame(edited, columns=['A','B','C','D','E','F','G','H'])
        st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'), "lottery.csv", "text/csv")
