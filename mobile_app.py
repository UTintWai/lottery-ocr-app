import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v61", layout="wide")
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
                # ဂဏန်းတွေ String ဖြစ်အောင် ' ထည့်ပေးခြင်းဖြင့် 002 ကို 2 ဖြစ်မသွားအောင်ကာကွယ်သည်
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
    # Parity check and detail improvement
    return easyocr.Reader(['en'], gpu=False)

def process_side_v61(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resolution ကို ၁၆၀၀ အထိ မြှင့်တင်ပြီး ပိုကြည်အောင်လုပ်သည်
    target_w = 1600 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # OCR ပိုမိအောင် Gray conversion နှင့် Denoising လုပ်ခြင်း
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    
    # Detail ဖတ်နှုန်းမြှင့်ထားသည်
    results = reader.readtext(gray, paragraph=False, mag_ratio=2.5, min_size=10)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]), 
            'y': np.mean([p[1] for p in bbox]), 
            'text': text
        })
    
    if not raw_data: return []

    # ROW GROUPING
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    if raw_data:
        curr_row = [raw_data[0]]
        for i in range(1, len(raw_data)):
            if raw_data[i]['y'] - curr_row[-1]['y'] < 30: # စာကြောင်းအမြင့်
                curr_row.append(raw_data[i])
            else:
                rows.append(curr_row)
                curr_row = [raw_data[i]]
        rows.append(curr_row)

    processed_side_data = []
    # 🎯 STRICT COLUMN BOUNDARIES (တိုင် ၄ တိုင်ကို ခွဲထုတ်ခြင်း)
    # ပုံတစ်ပုံကို ၄ ပိုင်း တိတိကျကျ ပိုင်းသည်
    cw = target_w / 4
    
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            # x coordinate အပေါ်မူတည်ပြီး ဘယ်တိုင်မှာ ရှိရမလဲဆိုတာ ဆုံးဖြတ်သည်
            col_idx = int(item['x'] // cw)
            if 0 <= col_idx < 4:
                txt = item['text'].upper().strip()
                # Ditto marks (", -, _, .) စတာတွေကို ရှာသည်
                if re.search(r'[။၊"=“_…\.\-]', txt) or (not txt.isdigit() and len(txt) == 1):
                    row_cells[col_idx] = "DITTO"
                else:
                    # အမှားများသော စာလုံးများကို ဂဏန်းအဖြစ်ပြောင်းသည်
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        # ဂဏန်းတိုင် (0, 2) ဖြစ်လျှင် ၃ လုံးစလုံးပေါ်အောင် လုပ်သည်
                        if col_idx % 2 == 0:
                            row_cells[col_idx] = num.zfill(3)[-3:]
                        else:
                            row_cells[col_idx] = num
        processed_side_data.append(row_cells)
    
    # Amount Columns (1, 3) အတွက် Ditto ဖြည့်ခြင်း
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_side_data)):
            v = str(processed_side_data[r][c]).strip()
            if v.isdigit() and v != "":
                last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_side_data[r][c] = last_val
            
    return processed_side_data

# --- UI ---
st.title("🔢 Lottery Pro v61 (Strict Alignment)")
st.info("တိုင်လွဲခြင်းနှင့် ဂဏန်းပျောက်ခြင်းများကို Pixel-level coordination ဖြင့် ပြင်ဆင်ထားပါသည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ဘယ် ၄ တိုင် (A,B,C,D)", type=['jpg', 'jpeg', 'png'])
with c2: up_right = st.file_uploader("ညာ ၄ တိုင် (E,F,G,H)", type=['jpg', 'jpeg', 'png'])

if st.button("🔍 Scan & Combine (8 Columns)"):
    left_side, right_side = [], []
    if up_left:
        img_l = cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1)
        left_side = process_side_v61(img_l)
    if up_right:
        img_r = cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1)
        right_side = process_side_v61(img_r)
            
    max_r = max(len(left_side), len(right_side))
    combined = []
    for i in range(max_r):
        l_row = left_side[i] if i < len(left_side) else ["","","",""]
        r_row = right_side[i] if i < len(right_side) else ["","","",""]
        combined.append(l_row + r_row)
    st.session_state['data_v61'] = combined

if 'data_v61' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    # အသုံးပြုသူ ကိုယ်တိုင် ပြင်နိုင်ရန်
    edited = st.data_editor(st.session_state['data_v61'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(edited):
            st.success("✅ Google Sheet ထဲသို့ ဒေတာများ ရောက်ရှိသွားပါပြီ!")
            st.balloons()
