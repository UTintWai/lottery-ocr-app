import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIGURATION ---
st.set_page_config(page_title="Lottery Pro v62", layout="wide")
SHEET_NAME = "LotteryData" 

def save_to_gsheet(data_to_save):
    try:
        if not data_to_save: return False
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).get_worksheet(0)
        
        formatted_data = []
        for row in data_to_save:
            # တိုင်တစ်ခုခုမှာ ဒေတာရှိမှ သိမ်းမည်
            if any(str(cell).strip() for cell in row):
                # Excel/Sheet 0 ပျောက်မသွားအောင် ' ခံသည်
                new_row = [f"'{str(cell)}" if str(cell).strip() != "" else "" for cell in row]
                formatted_data.append(new_row)
        
        if formatted_data:
            sheet.append_rows(formatted_data, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_side_v62(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resolution ကို တိကျအောင် 1600px မှာ ထိန်းထားသည်
    target_w = 1600 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR Resolution ပိုကောင်းအောင် mag_ratio တင်ထားသည်
    results = reader.readtext(gray, paragraph=False, mag_ratio=2.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]), 
            'y': np.mean([p[1] for p in bbox]), 
            'text': text
        })
    
    if not raw_data: return []

    # ROW GROUPING (စာကြောင်းများကို y coordinate အလိုက် စုစည်းသည်)
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 30: 
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    processed_table = []
    # 🎯 GRID SPLIT: ပုံတစ်ပုံစီကို ၄ တိုင် တိတိကျကျ ပိုင်းဖြတ်ခြင်း
    # Column Borders: [0-400], [401-800], [801-1200], [1201-1600]
    
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            x = item['x']
            # X coordinate အလိုက် သက်ဆိုင်ရာ ကော်လံထဲသို့ ထည့်သည်
            if x < 400: c_idx = 0
            elif x < 800: c_idx = 1
            elif x < 1200: c_idx = 2
            else: c_idx = 3
            
            txt = item['text'].upper().strip()
            # Ditto Marks ရှာဖွေခြင်း (", -, _, =, ။၊)
            if re.search(r'[။၊"=“_…\.\-\']', txt) or (not txt.isdigit() and len(txt) == 1):
                row_cells[c_idx] = "DITTO"
            else:
                # Digital-friendly replacement
                txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:] # ဂဏန်းတိုင်
                    else: row_cells[c_idx] = num # ထိုးကြေးတိုင်
        processed_table.append(row_cells)
    
    # 🔄 AUTO-FILL DITTO (ထိုးကြေးတိုင်များအတွက် အပေါ်က တန်ဖိုးယူသည်)
    for col in [1, 3]:
        last_val = ""
        for r in range(len(processed_table)):
            val = str(processed_table[r][col]).strip()
            if val.isdigit() and val != "":
                last_val = val
            elif (val == "DITTO" or val == "") and last_val != "":
                processed_table[r][col] = last_val
            
    return processed_table

# --- UI ---
st.title("🔢 Lottery Precision v62")
st.markdown("#### ဘယ်ပုံ/ညာပုံ ခွဲတင်ပါ။ တိုင်လွဲခြင်းကို ၁၀၀% ကာကွယ်ထားပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ပုံ (၁) - ဘယ် ၄ တိုင် (A,B,C,D)", type=['jpg', 'jpeg', 'png'])
with c2: up_right = st.file_uploader("ပုံ (၂) - ညာ ၄ တိုင် (E,F,G,H)", type=['jpg', 'jpeg', 'png'])

if st.button("🔍 Scan and Fix Columns"):
    left_data, right_data = [], []
    if up_left:
        img_l = cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1)
        left_data = process_side_v62(img_l)
    if up_right:
        img_r = cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1)
        right_data = process_side_v62(img_r)
            
    max_rows = max(len(left_data), len(right_data))
    final_combined = []
    for i in range(max_rows):
        l = left_data[i] if i < len(left_data) else ["","","",""]
        r = right_data[i] if i < len(right_data) else ["","","",""]
        final_combined.append(l + r)
    st.session_state['v62_data'] = final_combined

if 'v62_data' in st.session_state:
    st.subheader("စစ်ဆေးရန် ၈ တိုင် ဇယား")
    edited = st.data_editor(st.session_state['v62_data'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(edited):
            st.success("✅ Google Sheet ထဲသို့ ဒေတာများ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
            st.balloons()
