import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro (Auto-Fill Mode)", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models')

reader = load_ocr()

def process_and_fill(img, n_cols):
    # RAM Crash မဖြစ်အောင် Resize လုပ်ခြင်း
    h, w = img.shape[:2]
    img = cv2.resize(img, (1200, int(h * (1200 / w))))
    new_h, new_w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    
    data_list = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        data_list.append({'x': cx, 'y': cy, 'text': text.strip()})

    if not data_list: return []

    # --- ROW CLUSTERING (အတန်းခွဲခြင်း) ---
    data_list.sort(key=lambda k: k['y'])
    rows_list = []
    current_row = [data_list[0]]
    threshold = 30 # အတန်းအကွာအဝေး

    for i in range(1, len(data_list)):
        if data_list[i]['y'] - current_row[-1]['y'] < threshold:
            current_row.append(data_list[i])
        else:
            rows_list.append(current_row)
            current_row = [data_list[i]]
    rows_list.append(current_row)

    # --- GRID & AUTO-FILL LOGIC ---
    final_grid = []
    col_width = new_w / n_cols

    for row_data in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_data:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                txt = item['text'].upper()
                # Ditto သင်္ကေတများ စစ်ဆေးခြင်း
                if any(m in txt for m in ['"', '။', '=', 'U', 'V', '`', '4', '||', '11', 'LL']):
                    row_cells[c_idx] = "DITTO"
                else:
                    num = re.sub(r'[^0-9]', '', txt)
                    if num: row_cells[c_idx] = num.zfill(3) if len(num) <= 3 else num
        final_grid.append(row_cells)

    # --- SMART FILL-DOWN (အပေါ်ကဂဏန်း ကူးဖြည့်ခြင်း) ---
    # တစ်တိုင်ချင်းစီအတွက် အပေါ်ကနေ အောက်ကို စစ်မယ်
    for c in range(n_cols):
        last_val = ""
        for r in range(len(final_grid)):
            curr_val = final_grid[r][c].strip()
            
            # အကယ်၍ အကွက်က လွတ်နေရင် သို့မဟုတ် DITTO ဖြစ်နေရင်
            if curr_val == "" or curr_val == "DITTO":
                if last_val != "":
                    final_grid[r][c] = last_val # အပေါ်ကတန်ဖိုးကို ယူမယ်
            else:
                last_val = curr_val # တန်ဖိုးအသစ်တွေ့ရင် အမှတ်အသားလုပ်ထားမယ်
                
    return final_grid

def save_to_sheets(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").sheet1
        
        # Format for Google Sheets (Zeroes preservation)
        formatted = [[f"'{c}" if c != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Lottery Smart-Fill Pro (v5)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.info("အပေါ်ကဂဏန်းအတိုင်း အောက်ကအကွက်လွတ်တွေနဲ့ Ditto တွေကို အလိုအလျောက် ကူးဖြည့်ပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button("🔍 ဖတ်မယ် (Auto-Fill)"):
        with st.spinner("AI ကူးဖြည့်ပေးနေပါသည်..."):
            grid_res = process_and_fill(img, a_cols)
            st.session_state['data_v5'] = grid_res

if 'data_v5' in st.session_state:
    edited = st.data_editor(st.session_state['data_v5'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
