import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v8 (Final Ditto Fix)", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def process_v8(img, n_cols):
    h, w = img.shape[:2]
    # ပုံကို ပိုမိုရှင်းလင်းစေရန် contrast မြှင့်ခြင်း
    img_resized = cv2.resize(img, (1600, int(h * (1600 / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR results
    results = reader.readtext(gray, paragraph=False)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- ROW CLUSTERING (အတန်းခွဲခြင်း) ---
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 28 

    current_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - current_row[-1]['y'] < y_threshold:
            current_row.append(raw_data[i])
        else:
            rows_list.append(current_row)
            current_row = [raw_data[i]]
    rows_list.append(current_row)

    # --- GRID ASSIGNMENT ---
    final_grid = []
    img_w = gray.shape[1]
    col_width = img_w / n_cols

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                txt = item['text']
                # Ditto ဖြစ်နိုင်ခြေရှိသော သင်္ကေတများအားလုံးကို စစ်ဆေးခြင်း
                is_ditto = any(m in txt for m in ['"', '။', '=', '`', '||', '11', 'LL', 'V', '4', 'U', 'Y', '1', 'I', 'J', '(', ')'])
                
                if is_ditto and len(txt) <= 2: 
                    row_cells[c_idx] = "DITTO"
                else:
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: # ဂဏန်းတိုင် (၃ လုံး)
                            row_cells[c_idx] = num.zfill(3) if len(num) <= 3 else num[:3]
                        else: # ထိုးကြေးတိုင်
                            row_cells[c_idx] = num
        final_grid.append(row_cells)

    # --- 💡 SMART FILL LOGIC (အဓိကပြင်ဆင်ချက်) ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်များအတွက်သာ လုပ်ဆောင်မည်
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                
                # အကယ်၍ အကွက်က လွတ်နေရင် သို့မဟုတ် DITTO ဖြစ်နေရင်
                if val == "DITTO" or val == "":
                    if last_amt != "":
                        final_grid[r][c] = last_amt # အပေါ်ကထိုးကြေးကို ကူးထည့်ပေးမည်
                else:
                    last_amt = val # ဂဏန်းအသစ်တွေ့ရင် အမှတ်အသားပြုမည်
        else: # ၃ လုံးဂဏန်းတိုင်များအတွက်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = "" # ဂဏန်းတိုင်တွင် Ditto မရှိရပါ
                    
    return final_grid

def save_to_sheets_v8(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").sheet1
        
        formatted = [[f"'{c}" if c != "" else "" for c in row] for row in data if any(x != "" for x in row)]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Lottery Scanner v8 (Auto-Fill Fixed)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.info("ဗားရှင်း ၈ တွင် ထိုးကြေးတိုင်ရှိ အကွက်လွတ်များနှင့် Ditto များကို အပေါ်ဂဏန်းအတိုင်း အလိုအလျောက် ဖြည့်ပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=450)
    
    if st.button("🔍 Scan & Auto-Fill"):
        with st.spinner("အချက်အလက်များကို တိကျစွာ ဖြည့်သွင်းနေပါသည်..."):
            res = process_v8(img, a_cols)
            st.session_state['data_v8'] = res

if 'data_v8' in st.session_state:
    edited = st.data_editor(st.session_state['data_v8'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets_v8(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
