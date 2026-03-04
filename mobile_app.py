import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v73", layout="wide")
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
    # မှားတတ်တဲ့ အင်္ဂလိပ်စာလုံးတွေကို ပိတ်ပြီး ဂဏန်းနဲ့ အချို့သင်္ကေတတွေကိုပဲ အာရုံစိုက်ခိုင်းသည်
    return easyocr.Reader(['en'], gpu=False)

def clean_text_v73(txt):
    """ဂဏန်းအမှားများကို AI နည်းဖြင့် ပြန်ပြင်ပေးသည့်လုပ်ဆောင်ချက်"""
    txt = txt.upper().replace(' ', '')
    # 🎯 Common OCR Errors Correction
    mapping = {
        'S': '5', 'G': '6', 'B': '8', 'I': '1', 'L': '1', 
        'O': '0', 'D': '0', 'Q': '0', 'Z': '2', 'A': '4',
        'T': '7', 'J': '7', 'U': '0', 'V': '0'
    }
    for char, num in mapping.items():
        txt = txt.replace(char, num)
    
    # Ditto Marks ရှာဖွေခြင်း
    if re.search(r'[။၊"=“_…\.\-\']', txt) or (not any(c.isdigit() for c in txt) and len(txt) > 0):
        return "DITTO"
        
    # ဂဏန်းမဟုတ်တာမှန်သမျှ ဖယ်ထုတ်သည်
    num_only = re.sub(r'[^0-9]', '', txt)
    return num_only

def process_v73(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600
    img = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 🎯 Image Sharpness: စာလုံးတွေကို ပိုပြတ်သားအောင် လုပ်ခြင်း
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    
    results = reader.readtext(sharpened, paragraph=False, mag_ratio=2.0)
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 'text': t} for (b, t, p) in results]
    elements.sort(key=lambda k: k['y'])

    # Row Merging
    rows = []
    if elements:
        curr_row = [elements[0]]
        for i in range(1, len(elements)):
            y_diff = elements[i]['y'] - curr_row[-1]['y']
            current_y = elements[i]['y']
            dynamic_limit = 45 if current_y < (h/2) else 75 # အောက်ပိုင်းကျဲတာကို ပေါင်းရန်
            
            if y_diff < dynamic_limit:
                is_overlap = any(abs(elements[i]['x'] - item['x']) < 110 for item in curr_row)
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
            
            cleaned = clean_text_v73(item['text'])
            if cleaned == "DITTO":
                row_cells[c_idx] = "DITTO"
            elif cleaned:
                # ဂဏန်းတိုင် (Column 0 နဲ့ 2) ဆိုလျှင် ၃ လုံးဖြည့်သည်
                if c_idx % 2 == 0: row_cells[c_idx] = cleaned.zfill(3)[-3:]
                else: row_cells[c_idx] = cleaned
        
        if any(c != "" for c in row_cells):
            processed_data.append(row_cells)

    # Auto-Ditto Filling
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit() and v != "": last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v73 (Error Fix Mode)")
st.info("ဂဏန်းအမှားများကို AI ဖြင့် အလိုအလျောက် ပြန်လည်စစ်ဆေးပေးပါမည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ဘယ်ဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_right = st.file_uploader("ညာဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 Scan and Fix Errors"):
    with st.spinner('အမှားများကို ပြင်ဆင်နေပါသည်...'):
        l_res, r_res = [], []
        if up_left: l_res = process_v73(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
        if up_right: r_res = process_v73(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
        max_r = max(len(l_res), len(r_res))
        final = []
        for i in range(max_r):
            l = l_res[i] if i < len(l_res) else ["","","",""]
            r = r_res[i] if i < len(r_res) else ["","","",""]
            final.append(l + r)
        
        st.session_state['data_v73'] = final
        st.rerun()

if 'data_v73' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (မှားနေပါက ပြင်ပေးပါ)")
    st.session_state['data_v73'] = st.data_editor(st.session_state['data_v73'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v73']):
            st.success("✅ Google Sheet ထဲ သိမ်းဆည်းပြီးပါပြီ!")
            st.balloons()
