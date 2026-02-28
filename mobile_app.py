import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gc
import gspread
import time
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Scanner v34", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v34(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        status = st.empty()
        for i, row in enumerate(data):
            # တိုင် ၈ တိုင်ထက် မပိုအောင် အသေဖြတ်ချမည်
            clean_row = row[:8]
            if any(str(c).strip() for c in clean_row):
                formatted = [f"'{str(c)}" if str(c).strip() != "" else "" for c in clean_row]
                sheet.append_row(formatted)
                status.text(f"သိမ်းဆည်းနေပါသည်... ({i+1}/{len(data)})")
                time.sleep(0.3)
        return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 8-Column Strict Alignment v34")

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v34(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1200 # ဗောက်ချာကို ၁၂၀၀ pixel width သတ်မှတ်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ပုံကို အပိုင်း ၈ ပိုင်းခွဲဖတ်မည် (RAM ချွေတာရန်)
    all_results = []
    num_parts = 8
    for i in range(num_parts):
        y1 = max(0, int(gray.shape[0] * (i/num_parts)) - 30)
        y2 = min(gray.shape[0], int(gray.shape[0] * ((i+1)/num_parts)) + 30)
        res = reader.readtext(gray[y1:y2, :], paragraph=False)
        for (bbox, text, prob) in res:
            all_results.append({
                'x': np.mean([p[0] for p in bbox]),
                'y': np.mean([p[1] for p in bbox]) + y1,
                'text': text
            })
    
    if not all_results: return []

    # ROW CLUSTERING
    all_results.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [all_results[0]]
    for i in range(1, len(all_results)):
        if all_results[i]['y'] - curr_row[-1]['y'] < 25:
            curr_row.append(all_results[i])
        else:
            rows.append(curr_row)
            curr_row = [all_results[i]]
    rows.append(curr_row)

    # --- STRICT 8-COLUMN LOGIC ---
    # တိုင်တစ်ခုချင်းစီ၏ နယ်နိမိတ်ကို အသေသတ်မှတ်ခြင်း
    # Column 0: 0-150, Col 1: 151-300, ... Col 7: 1050-1200
    final_data = []
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            # စာလုံး၏ x-coordinate ကိုကြည့်ပြီး တိုင်ခွဲခြင်း
            c_idx = int(item['x'] // 150) 
            if 0 <= c_idx < 8:
                txt = re.sub(r'[^0-9"။=LVUYI/]', '', item['text'].upper())
                row_cells[c_idx] = (row_cells[c_idx] + txt).strip()
        
        # Data Formatting
        for c in range(8):
            val = row_cells[c]
            if c % 2 == 0 and val.isdigit():
                row_cells[c] = val.zfill(3)[:3]
            elif any(m in val for m in ['"', '။', '=', 'L', 'V', 'U', 'Y', 'I', '/']):
                row_cells[c] = "DITTO"
        final_data.append(row_cells)

    # Amount Fill Down
    for c in [1, 3, 5, 7]:
        last_val = ""
        for r in range(len(final_data)):
            if final_data[r][c] == "DITTO" and last_val: final_data[r][c] = last_val
            elif final_data[r][c].isdigit(): last_val = final_data[r][c]
            
    # Number Clean up
    for c in [0, 2, 4, 6]:
        for r in range(len(final_data)):
            if final_data[r][c] == "DITTO": final_data[r][c] = ""
            
    return final_data

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=500)
    
    if st.button("🔍 ၈ တိုင် တိတိကျကျ Scan ဖတ်မည်"):
        with st.spinner("တိုင် ၈ တိုင်ကို ညီအောင် ညှိနေပါသည်..."):
            res = process_v34(img)
            st.session_state['data_v34'] = res

if 'data_v34' in st.session_state:
    # Table မှာ တိုင် ၈ တိုင်ပဲပြရန် ကန့်သတ်ခြင်း
    edited = st.data_editor(st.session_state['data_v34'], use_container_width=True)
    if st.button("💾 Google Sheet ထဲ သိမ်းမည်"):
        if save_to_sheets_v34(edited):
            st.success("Sheet ထဲသို့ ၈ တိုင်စလုံး အညီအညာ ဝင်သွားပါပြီ!")
