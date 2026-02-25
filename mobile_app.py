import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner Logic Fix", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models')

reader = load_ocr()

def process_logic_fix(img, n_cols):
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
    threshold = 28 # လက်ရေးအနိမ့်အမြင့်ပေါ်မူတည်၍ ညှိထားသည်

    for i in range(1, len(data_list)):
        if data_list[i]['y'] - current_row[-1]['y'] < threshold:
            current_row.append(data_list[i])
        else:
            rows_list.append(current_row)
            current_row = [data_list[i]]
    rows_list.append(current_row)

    # --- GRID CALCULATION ---
    final_grid = []
    col_width = new_w / n_cols

    for row_data in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_data:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                txt = item['text'].upper()
                # Ditto စစ်ဆေးခြင်း
                if any(m in txt for m in ['"', '။', '=', '`', '||', '11', 'LL', 'V']):
                    row_cells[c_idx] = "DITTO"
                else:
                    # ဂဏန်းများကို သီးသန့်ထုတ်ယူခြင်း
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        # ၃ လုံးဂဏန်း (Column 0, 2, 4, 6) အတွက် zero padding
                        if c_idx % 2 == 0:
                            row_cells[c_idx] = num.zfill(3) if len(num) <= 3 else num
                        else:
                            row_cells[c_idx] = num
        final_grid.append(row_cells)

    # --- COLUMN-WISE DITTO LOGIC (ထိုးကြေးအတွက်သာ) ---
    for c in range(n_cols):
        # ထိုးကြေးတိုင်များဖြစ်သော Column 1, 3, 5, 7 တွင်သာ Ditto ကူးယူမည်
        if c % 2 != 0: 
            for r in range(1, len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = final_grid[r-1][c]
                # ဂဏန်း ၃ လုံးတိုင် (c % 2 == 0) များတွင် DITTO တွေ့ပါက 
                # ဂဏန်းမကူးဘဲ အလွတ်သာထားမည် သို့မဟုတ် Ditto အတိုင်းထားမည်
        else:
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = "" # ဂဏန်းတိုင်တွင် Ditto မရှိရပါ

    return final_grid

def save_to_sheets(data):
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
st.title("🔢 Lottery Precision Fix (Myanmar Logic)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.markdown("""
    **ပြင်ဆင်ထားချက်များ:**
    * ၃ လုံးဂဏန်းတိုင်များတွင် အပေါ်အောက် မကူးတော့ပါ။
    * ထိုးကြေးတိုင်များတွင်သာ `။` (Ditto) ပါက အပေါ်မှ ကူးယူပါမည်။
    """)

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button("🔍 တိကျစွာ ဖတ်ရှုမည်"):
        with st.spinner("Logic ပြန်လည်စစ်ဆေးနေပါသည်..."):
            res = process_logic_fix(img, a_cols)
            st.session_state['data_fixed'] = res

if 'data_fixed' in st.session_state:
    edited = st.data_editor(st.session_state['data_fixed'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets(edited):
            st.success("ဒေတာများ အမှန်ကန်ဆုံး သိမ်းဆည်းပြီးပါပြီ!")
