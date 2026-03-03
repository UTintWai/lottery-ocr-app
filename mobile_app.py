import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v59", layout="wide")
SHEET_NAME = "LotteryData" 

# --- GOOGLE SHEETS SAVE FUNCTION ---
def save_to_gsheet(data_to_save):
    try:
        if not data_to_save or len(data_to_save) == 0:
            st.warning("ပို့ရန် ဒေတာမရှိပါဗျ။")
            return False
            
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        if "gcp_service_account" not in st.secrets:
            st.error("Secrets ထဲမှာ Credentials မရှိသေးပါဗျ။")
            return False
            
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Sheet ကိုဖွင့်သည်
        sh = client.open(SHEET_NAME)
        sheet = sh.get_worksheet(0) # ပထမဆုံး sheet ကို ယူသည်
        
        # Data တွေကို String format ပြောင်းခြင်း (0 ပျောက်မသွားစေရန်)
        formatted_data = []
        for row in data_to_save:
            # အကွက်အလွတ် မဟုတ်မှသာ ထည့်မည်
            if any(str(cell).strip() for cell in row):
                new_row = [f"'{str(cell)}" if str(cell).strip() != "" else "" for cell in row]
                formatted_data.append(new_row)
        
        if formatted_data:
            sheet.append_rows(formatted_data, value_input_option='USER_ENTERED')
            return True
        else:
            st.warning("ဇယားထဲမှာ ဒေတာအလွတ်တွေပဲ ဖြစ်နေပါတယ်ဗျ။")
            return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

# --- OCR ENGINE ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_side(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1500 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Image Enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    results = reader.readtext(thresh, paragraph=False)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({'x': np.mean([p[0] for p in bbox]), 'y': np.mean([p[1] for p in bbox]), 'text': text})
    
    if not raw_data: return []

    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 25: curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    processed_side_data = []
    col_width = target_w / 4 
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 4:
                txt = item['text'].upper().strip()
                if re.search(r'[။၊"=“_…\.\-]', txt): row_cells[c_idx] = "DITTO"
                else:
                    txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: row_cells[c_idx] = num.zfill(3)[-3:]
                        else: row_cells[c_idx] = num
        processed_side_data.append(row_cells)
    
    # Ditto fill
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_side_data)):
            v = str(processed_side_data[r][c])
            if v.isdigit() and v != "": last_val = v
            elif (v == "DITTO" or v == "") and last_val != "": processed_side_data[r][c] = last_val
            
    return processed_side_data

# --- UI ---
st.title("🔢 Side-by-Side Expert v59")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ဘယ် ၄ တိုင် (A,B,C,D)", type=['jpg', 'png'])
with c2: up_right = st.file_uploader("ညာ ၄ တိုင် (E,F,G,H)", type=['jpg', 'png'])

if st.button("🔍 Scan and Combine"):
    left_data, right_data = [], []
    if up_left:
        img_l = cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1)
        left_data = process_side(img_l)
    if up_right:
        img_r = cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1)
        right_data = process_side(img_r)
            
    max_rows = max(len(left_data), len(right_data))
    final_8_cols = []
    for i in range(max_rows):
        l_row = left_data[i] if i < len(left_data) else ["","","",""]
        r_row = right_data[i] if i < len(right_data) else ["","","",""]
        final_8_cols.append(l_row + r_row)
        
    st.session_state['combined_data'] = final_8_cols

# --- DATA TABLE & SAVE ---
if 'combined_data' in st.session_state:
    st.subheader("ပေါင်းစပ်ပြီး ၈ တိုင် ဇယား")
    # အရေးကြီးသည်- ဒေတာကို တိုက်ရိုက် edit နိုင်အောင် လုပ်ထားသည်
    edited_data = st.data_editor(st.session_state['combined_data'], use_container_width=True, num_rows="dynamic", key="editor")
    
    if st.button("💾 Save to Google Sheet"):
        with st.spinner("Sheet ထဲသို့ ပို့နေပါသည်..."):
            # editor ကနေ တိုက်ရိုက်ဒေတာကို ယူပြီး ပို့သည်
            if save_to_gsheet(edited_data):
                st.success("✅ Google Sheet ထဲသို့ အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
                st.balloons()
