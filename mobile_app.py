import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v63", layout="wide")
SHEET_NAME = "LotteryData" 

def save_to_gsheet(data_to_save):
    try:
        if not data_to_save: return False
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).get_worksheet(0)
        
        formatted_rows = []
        for row in data_to_save:
            # ဂဏန်းရှေ့က 0 မပျောက်စေရန် ' ခံပြီး သိမ်းခြင်း
            formatted_rows.append([f"'{str(c)}" if str(c).strip() != "" else "" for c in row])
        
        if formatted_rows:
            sheet.append_rows(formatted_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_side_v63(img):
    reader = load_ocr()
    # ပုံကို ပိုပြတ်သားအောင် Resolution မြှင့်ခြင်း
    h, w = img.shape[:2]
    target_w = 1800 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # Contrast မြှင့်တင်ရေး (ဂဏန်းအစက်အပြောက်လေးတွေပါ မိစေရန်)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.convertScaleAbs(gray, alpha=1.2, beta=10)
    
    # OCR Scan (Detail Mode)
    results = reader.readtext(enhanced, paragraph=False, mag_ratio=2.5)
    
    raw_elements = []
    for (bbox, text, prob) in results:
        raw_elements.append({
            'x': np.mean([p[0] for p in bbox]), 
            'y': np.mean([p[1] for p in bbox]), 
            'text': text
        })
    
    if not raw_elements: return []

    # စာကြောင်းခွဲခြင်း (Row Grouping)
    raw_elements.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_elements[0]]
    for i in range(1, len(raw_elements)):
        if raw_elements[i]['y'] - curr_row[-1]['y'] < 35: 
            curr_row.append(raw_elements[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_elements[i]]
    rows.append(curr_row)

    processed_data = []
    # 🎯 VIRTUAL GRID LOCK (တိုင် ၄ တိုင်ကို Pixel အတိအကျ ပိုင်းခြားခြင်း)
    # တိုင်တစ်ခုချင်းစီ၏ နယ်နိမိတ် [0-450, 451-900, 901-1350, 1351-1800]
    col_width = target_w / 4
    
    for r_items in rows:
        cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 4:
                txt = item['text'].upper().strip()
                # Ditto marks (", -, _, .) များကို ရှာဖွေခြင်း
                if re.search(r'[။၊"=“_…\.\-]', txt):
                    cells[c_idx] = "DITTO"
                else:
                    # မှားတတ်သော စာလုံးများကို ဂဏန်းအဖြစ် ပြောင်းလဲခြင်း
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: cells[c_idx] = num.zfill(3)[-3:] # ဂဏန်းတိုင် (၃ လုံးပြည့်)
                        else: cells[c_idx] = num # ထိုးကြေးတိုင်
        processed_data.append(cells)
    
    # ထိုးကြေးတိုင်များအတွက် Auto-fill (DITTO)
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit() and v != "": last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
            
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v63 (Final Alignment)")
st.info("တိုင်လွဲခြင်းနှင့် ဂဏန်းပျောက်ခြင်းများကို Pixel-based mapping ဖြင့် ပြင်ဆင်ထားပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ပုံ (၁) - ဘယ် ၄ တိုင်", type=['jpg', 'png', 'jpeg'])
with c2: up_right = st.file_uploader("ပုံ (၂) - ညာ ၄ တိုင်", type=['jpg', 'png', 'jpeg'])

if st.button("🔍 Scan All Columns"):
    l_res, r_res = [], []
    if up_left:
        l_res = process_side_v63(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
    if up_right:
        r_res = process_side_v63(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
    max_r = max(len(l_res), len(r_res))
    final = []
    for i in range(max_r):
        row_l = l_res[i] if i < len(l_res) else ["","","",""]
        row_r = r_res[i] if i < len(r_res) else ["","","",""]
        final.append(row_l + row_r)
    st.session_state['v63_data'] = final

if 'v63_data' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (A မှ H)")
    edited = st.data_editor(st.session_state['v63_data'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(edited):
            st.success("✅ ဒေတာများကို Sheet ထဲသို့ တိုင်မှန်ကန်စွာ ပို့ဆောင်ပြီးပါပြီ!")
            st.balloons()
