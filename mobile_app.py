import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner Ultimate", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models')

reader = load_ocr()

def process_smart_grid(img, n_cols):
    # RAM မသေစေရန် ပုံကို သင့်တော်သလို လျှော့ချခြင်း
    h, w = img.shape[:2]
    img = cv2.resize(img, (1200, int(h * (1200 / w))))
    new_h, new_w = img.shape[:2]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)
    
    # စာလုံးအားလုံးကို နေရာ (x, y) အလိုက် စာရင်းပြုစုခြင်း
    data_list = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        data_list.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not data_list: return [["" for _ in range(n_cols)]]

    # --- ROW CLUSTERING ---
    # y-coordinate တူရာစုပြီး အတန်းခွဲခြင်း (စာလုံးစောင်းနေမှုကို ဖြေရှင်းရန်)
    data_list.sort(key=lambda k: k['y'])
    rows_list = []
    current_row = [data_list[0]]
    threshold = 25  # အတန်းတစ်ခုနဲ့တစ်ခုကြား ကွာခြားချက် (လိုအပ်ရင် ချိန်ညှိနိုင်သည်)

    for i in range(1, len(data_list)):
        if data_list[i]['y'] - current_row[-1]['y'] < threshold:
            current_row.append(data_list[i])
        else:
            rows_list.append(current_row)
            current_row = [data_list[i]]
    rows_list.append(current_row)

    # --- COLUMN ASSIGNMENT ---
    final_grid = []
    col_width = new_w / n_cols

    for row_data in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_data:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                # DITTO Logic
                if any(m in item['text'] for m in ['"', '။', '=', 'U', 'V', '`', '4', '||']):
                    row_cells[c_idx] = "DITTO"
                else:
                    num = re.sub(r'[^0-9]', '', item['text'])
                    if num: row_cells[c_idx] = num.zfill(3)
        final_grid.append(row_cells)

    # DITTO Fill logic
    for c in range(n_cols):
        for r in range(1, len(final_grid)):
            if final_grid[r][c] == "DITTO" and final_grid[r-1][c] != "":
                final_grid[r][c] = final_grid[r-1][c]
                
    return final_grid

def save_to_sheets(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").sheet1
        
        # Google Sheet မှာ 0 မပျောက်အောင် Quote ခံပြီး ပို့ခြင်း
        formatted = [[f"'{c}" if c != "" else "" for c in row] for row in data if any(x != "" for x in row)]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Lottery Pro Scanner (Anti-Shift Mode)")

with st.sidebar:
    st.header("Settings")
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.warning("မှတ်ချက် - လက်ရေးဗောက်ချာများကို တည့်တည့်ရိုက်လေ ပိုတိကျလေဖြစ်ပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button("🔍 ဒေတာဖတ်မယ်"):
        with st.spinner("ဖတ်နေပါသည်..."):
            grid_res = process_smart_grid(img, a_cols)
            st.session_state['data_v4'] = grid_res

if 'data_v4' in st.session_state:
    edited = st.data_editor(st.session_state['data_v4'], use_container_width=True)
    if st.button("💾 Google Sheet သို့ ပို့မည်"):
        if save_to_sheets(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
