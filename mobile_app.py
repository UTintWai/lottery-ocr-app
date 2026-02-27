import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v24", layout="wide")

@st.cache_resource
def load_ocr():
    # လက်ရေးအတွက် link_threshold ကို အကောင်းဆုံးအနေအထား 0.4 ထားပါမည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def save_to_sheets_v24(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        # 062 ကဲ့သို့သော ဂဏန်းများအတွက် formatting သေချာစေရန်
        formatted = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Lottery Scanner v24 (Precision Focus)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.warning("V24: လက်ရေး Ditio (။) များကို ပိုမိုတိကျစွာ ခွဲခြားနိုင်အောင် ပြုပြင်ထားပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v24(img, n_cols):
    h, w = img.shape[:2]
    target_w = 1400 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # လက်ရေးအတွက် အသေးစိတ်ဖတ်ရန် settings
    results = reader.readtext(gray, paragraph=False, link_threshold=0.4, mag_ratio=1.5, min_size=10)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper(), 'prob': prob})

    if not raw_data: return []

    # ROW CLUSTERING
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

    col_edges = np.linspace(0, target_w, n_cols + 1)
    final_grid = []

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        bins = [[] for _ in range(n_cols)]
        
        for item in row_items:
            c_idx = int(np.searchsorted(col_edges, item['x']) - 1)
            if 0 <= c_idx < n_cols: bins[c_idx].append(item)
        
        for c in range(n_cols):
            bins[c].sort(key=lambda k: k['x'])
            combined_txt = "".join([i['text'] for i in bins[c]])
            
            # --- IMPROVED DITTO RECOGNITION ---
            # လက်ရေး "။" ကို AI မှ 4, 11, U, V, 11, / စသည်ဖြင့် မှားဖတ်လေ့ရှိသည်ကို logic ဖြင့်စစ်မည်
            # အကယ်၍ စာသားသည် တိုပြီး အောက်ပါ pattern များထဲပါက Ditto ဟု ယူဆမည်
            is_ditto_pattern = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '11', '/', '(', ')', 'I'])
            
            if is_ditto_pattern and len(combined_txt) <= 2:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: # ဂဏန်းတိုင် (၃ လုံးဖြစ်စေရန်)
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေးတိုင်
                        row_cells[c] = num
        final_grid.append(row_cells)

    # --- ADVANCED AUTO-FILL ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                # အကယ်၍ အကွက်လွတ်နေလျှင် သို့မဟုတ် Ditto ဖြစ်လျှင် အပေါ်ကတန်ဖိုးယူမည်
                if val in ["DITTO", ""]:
                    if last_amt: final_grid[r][c] = last_amt
                else:
                    last_amt = val
        else: # ဂဏန်းတိုင်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=500)
    
    if st.button(f"🔍 Scan {a_cols} Columns"):
        with st.spinner("လက်ရေးများကို သေချာစွာ စစ်ဆေးနေပါသည်..."):
            res = process_v24(img, a_cols)
            st.session_state['data_v24'] = res

if 'data_v24' in st.session_state:
    edited = st.data_editor(st.session_state['data_v24'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets_v24(edited):
            st.success("Sheet ထဲသို့ ဒေတာများ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
