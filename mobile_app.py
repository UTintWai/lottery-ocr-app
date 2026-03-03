import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v60", layout="wide")
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
            if any(str(cell).strip() for cell in row):
                # ဂဏန်းရှေ့က 0 မပျောက်အောင် ' ခံပေးခြင်း
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

def process_side_v60(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600 # ပိုကြည်အောင် width မြှင့်လိုက်သည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # စာလုံးတွေကို ပိုပြတ်သားအောင် Contrast မြှင့်ခြင်း
    enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Scan with higher detail
    results = reader.readtext(enhanced, paragraph=False, mag_ratio=2.0)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]), 
            'y': np.mean([p[1] for p in bbox]), 
            'text': text
        })
    
    if not raw_data: return []

    # ROW GROUPING (စာကြောင်းခွဲခြင်း)
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 28: # အကွာအဝေးကို ညှိထားသည်
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    processed_side_data = []
    # --- 🎯 COLUMN GRID LOCK (တိုင် ၄ တိုင်ကို တိတိကျကျ ပိုင်းခြားခြင်း) ---
    # တိုင်တစ်ခုချင်းစီရဲ့ အကျယ်ကို ၄ ပုံ ၁ ပုံ ပိုင်းလိုက်သည်
    col_bounds = [0, 400, 800, 1200, 1600] 
    
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            x_val = item['x']
            # ဘယ်တိုင်ထဲမှာ ရှိလဲဆိုတာကို x_val နဲ့ စစ်သည်
            for c in range(4):
                if col_bounds[c] <= x_val < col_bounds[c+1]:
                    txt = item['text'].upper().strip()
                    # Ditto Detection
                    if re.search(r'[။၊"=“_…\.\-]', txt):
                        row_cells[c] = "DITTO"
                    else:
                        txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                        num = re.sub(r'[^0-9]', '', txt)
                        if num:
                            if c % 2 == 0: row_cells[c] = num.zfill(3)[-3:] # ဂဏန်းတိုင်
                            else: row_cells[c] = num # ထိုးကြေးတိုင်
        processed_side_data.append(row_cells)
    
    # Auto-fill Ditto for amounts
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_side_data)):
            v = str(processed_side_data[r][c]).strip()
            if v.isdigit() and v != "": last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_side_data[r][c] = last_val
            
    return processed_side_data

# --- UI ---
st.title("🔢 Lottery precision v60")
st.markdown("#### ဘယ်ပုံ/ညာပုံ ခွဲတင်ပါ။ တိုင်လွဲခြင်းနှင့် ဂဏန်းပျောက်ခြင်းကို ပြင်ဆင်ထားပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ဘယ် ၄ တိုင် (A,B,C,D)", type=['jpg', 'png'])
with c2: up_right = st.file_uploader("ညာ ၄ တိုင် (E,F,G,H)", type=['jpg', 'png'])

if st.button("🔍 Scan and Fix Alignment"):
    left_data, right_data = [], []
    if up_left:
        img_l = cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1)
        left_data = process_side_v60(img_l)
    if up_right:
        img_r = cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1)
        right_data = process_side_v60(img_r)
            
    max_rows = max(len(left_data), len(right_data))
    final_8_cols = []
    for i in range(max_rows):
        l_row = left_data[i] if i < len(left_data) else ["","","",""]
        r_row = right_data[i] if i < len(right_data) else ["","","",""]
        final_8_cols.append(l_row + r_row)
    st.session_state['combined_v60'] = final_8_cols

if 'combined_v60' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (A မှ H)")
    edited_data = st.data_editor(st.session_state['combined_v60'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(edited_data):
            st.success("✅ Sheet ထဲသို့ ဒေတာများ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
            st.balloons()
