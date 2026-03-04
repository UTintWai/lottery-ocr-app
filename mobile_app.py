import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v74", layout="wide")
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
    # Allowlist သုံးပြီး ဂဏန်းတွေကိုပဲ အဓိကဖတ်ခိုင်းမည်
    return easyocr.Reader(['en'], gpu=False)

def preprocess_image(img):
    """ပုံရိပ်ကို အဆင့်မြင့်သန့်စင်ခြင်း"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ၁။ အလင်းအမှောင် ညှိခြင်း (Adaptive Thresholding)
    # စက္ကူအရောင်ကို အဖြူ၊ စာလုံးကို အမည်း အတိအကျခွဲထုတ်သည်
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # ၂။ စာလုံးများကို ပိုမိုထင်ရှားစေရန် Dilation လုပ်ခြင်း
    kernel = np.ones((1, 1), np.uint8)
    processed = cv2.dilate(thresh, kernel, iterations=1)
    
    return processed

def clean_and_format_v74(txt):
    """စာသားများကို ဂဏန်းအဖြစ် ပြောင်းလဲသန့်စင်ခြင်း"""
    txt = txt.upper().replace(' ', '')
    # 🎯 OCR အမှားပြင်ဆင်ချက်များ
    mapping = {
        'S': '5', 'G': '6', 'B': '8', 'I': '1', 'L': '1', '|': '1',
        'O': '0', 'D': '0', 'Q': '0', 'Z': '2', 'A': '4', 'T': '7'
    }
    for char, num in mapping.items():
        txt = txt.replace(char, num)
        
    if re.search(r'[။၊"=“_…\.\-\']', txt):
        return "DITTO"
        
    num_only = re.sub(r'[^0-9]', '', txt)
    return num_only

def process_v74(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1800 # Resolution မြှင့်သည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # ပုံကို အရင် သန့်စင်မည်
    processed_img = preprocess_image(img_resized)
    
    # OCR Scan (ဂဏန်းများကိုပဲ အဓိကထားဖတ်ရန်)
    results = reader.readtext(processed_img, paragraph=False, mag_ratio=2.5)
    
    if not results: return []

    elements = [{'x': np.mean([p[0] for p in b]), 'y': np.mean([p[1] for p in b]), 'text': t} for (b, t, p) in results]
    elements.sort(key=lambda k: k['y'])

    # Row Grouping Logic
    rows = []
    if elements:
        curr_row = [elements[0]]
        for i in range(1, len(elements)):
            y_diff = elements[i]['y'] - curr_row[-1]['y']
            # အောက်ပိုင်းကျဲတာကို ဖြေရှင်းရန် Dynamic Threshold
            limit = 45 if elements[i]['y'] < (len(processed_img)/2) else 80
            
            if y_diff < limit:
                # Column ထပ်မသွားအောင် check သည်
                if not any(abs(elements[i]['x'] - item['x']) < 120 for item in curr_row):
                    curr_row.append(elements[i])
                else:
                    rows.append(curr_row); curr_row = [elements[i]]
            else:
                rows.append(curr_row); curr_row = [elements[i]]
        rows.append(curr_row)

    processed_data = []
    col_width = target_w / 4
    for r_items in rows:
        row_cells = ["" for _ in range(4)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if c_idx > 3: c_idx = 3
            
            val = clean_and_format_v74(item['text'])
            if val == "DITTO":
                row_cells[c_idx] = "DITTO"
            elif val:
                if c_idx % 2 == 0: row_cells[c_idx] = val.zfill(3)[-3:]
                else: row_cells[c_idx] = val
        
        if any(c != "" for c in row_cells):
            processed_data.append(row_cells)

    # Smart Ditto Fill
    for c in [1, 3]:
        last_val = ""
        for r in range(len(processed_data)):
            v = str(processed_data[r][c]).strip()
            if v.isdigit(): last_val = v
            elif (v == "DITTO" or v == "") and last_val != "":
                processed_data[r][c] = last_val
                
    return processed_data

# --- UI ---
st.title("🔢 Lottery Pro v74 (HD Scan Mode)")
st.info("Binarization နည်းပညာဖြင့် စာလုံးများကို ပိုမိုပြတ်သားစွာ ဖတ်ရှုပေးပါမည်။")

c1, c2 = st.columns(2)
with c1: up_left = st.file_uploader("ဘယ်ဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_right = st.file_uploader("ညာဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 High-Precision Scan"):
    with st.spinner('ပုံရိပ်ကို အဆင့်မြှင့်တင်ပြီး ဖတ်နေပါသည်...'):
        l_res, r_res = [], []
        if up_left: l_res = process_v74(cv2.imdecode(np.frombuffer(up_left.read(), np.uint8), 1))
        if up_right: r_res = process_v74(cv2.imdecode(np.frombuffer(up_right.read(), np.uint8), 1))
            
        max_r = max(len(l_res), len(r_res))
        final = []
        for i in range(max_r):
            l = l_res[i] if i < len(l_res) else ["","","",""]
            r = r_res[i] if i < len(r_res) else ["","","",""]
            final.append(l + r)
        
        st.session_state['data_v74'] = final
        st.rerun()

if 'data_v74' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    st.session_state['data_v74'] = st.data_editor(st.session_state['data_v74'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v74']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!"); st.balloons()
