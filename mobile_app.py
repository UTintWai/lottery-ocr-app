import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery Pro v49", layout="wide")

@st.cache_resource
def load_ocr():
    # GPU မရှိသော PC များတွင် မြန်ဆန်စေရန်
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v49(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"] 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        table_rows = []
        for row in data:
            if any(str(c).strip() for c in row[:8]):
                # '062' ကဲ့သို့ ပေါ်ရန် single quote ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v49(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # Resolution ကို ပိုတိုးလိုက်သည် (2500px) - စာလုံးသေးများအတွက်
    target_w = 2500 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # --- 📸 DEEP CONTRAST ENHANCEMENT (အလင်းအမှောင်ကို အနက်ရှိုင်းဆုံးညှိခြင်း) ---
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)
    enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # OCR Scan
    results = reader.readtext(enhanced, paragraph=False)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text
        })

    if not raw_data: return []

    # ROW GROUPING (စာကြောင်းခွဲခြင်း logic ကို ပိုကျပ်လိုက်သည်)
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 25: 
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    # --- 8-COLUMN GRID MAPPING ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                txt = item['text'].upper().strip()
                # Ditto mark စစ်ဆေးခြင်း
                if re.search(r'[။၊"=“_…\.\-]', txt) or len(txt) == 1:
                    if not txt.isdigit(): row_cells[c_idx] = "DITTO"
                
                # --- 🔥 CHARACTER CORRECTION LOGIC (စာလုံးပြုပြင်ခြင်း) ---
                # မှားတတ်သောစာလုံးများကို Structure အလိုက် ပြင်ဆင်ခြင်း
                # (S -> 5, G -> 6, I -> 1, B -> 8, O -> 0, A -> 4, Z -> 2)
                txt = txt.replace('S','5').replace('G','6').replace('I','1').replace('B','8').replace('O','0').replace('A','4').replace('Z','2')
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    # နံပါတ်တိုင် (0,2,4,6) ဆိုလျှင် ၃ လုံးဖြစ်အောင် ဖြည့်မည်
                    if c_idx % 2 == 0:
                        row_cells[c_idx] = num.zfill(3)[-3:]
                    else:
                        row_cells[c_idx] = num
        final_table.append(row_cells)

    # --- DITTO & BLANK FILL (အပေါ်ဆုံးကနေ အောက်ထိ ဇွတ်ဖြည့်မည်) ---
    for c in [1, 3, 5, 7]: # ထိုးကြေးတိုင်များ
        active_amt = ""
        for r in range(len(final_table)):
            val = str(final_table[r][c]).strip()
            if val.isdigit() and val != "":
                active_amt = val
            elif (val == "DITTO" or val == "") and active_amt != "":
                final_table[r][c] = active_amt

    return final_table

# --- UI ---
st.title("🔢 Lottery Pro v49 (Accuracy Fix)")
st.warning("မှတ်ချက်: ဤ Version သည် ၃ နဲ့ ၈ ၊ ၅ နဲ့ ၆ တို့ကို ပိုမိုတိကျစွာ ခွဲခြားနိုင်ရန် အဆင့်မြှင့်တင်ထားပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ (အကြည်ဆုံးပုံကို သုံးပါ)", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 Scan All 8 Columns"):
        with st.spinner("အမှားနည်းအောင် အသေးစိတ် စကင်ဖတ်နေပါသည်..."):
            res = process_v49(img)
            st.session_state['data_v49'] = res

if 'data_v49' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    # Edit လုပ်နိုင်သော ဇယား
    edited = st.data_editor(st.session_state['data_v49'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v49(edited):
            st.success("Google Sheet ထဲသို့ ပို့လိုက်ပါပြီ!")
