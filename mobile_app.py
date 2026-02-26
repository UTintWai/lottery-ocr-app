import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v23", layout="wide")

@st.cache_resource
def load_ocr():
    # link_threshold မြှင့်ထားသဖြင့် ဘေးချင်းကပ်ဂဏန်းများ ပေါင်းဖတ်မည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def save_to_sheets_v23(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        # Google Sheet မှာ 003 ပေါ်စေရန် ' ခံပေးခြင်း
        formatted = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Lottery Scanner v23 (Ultimate Precision)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.info("V23: ၃ လုံးဂဏန်းနှင့် ထိုးကြေးတွဲဖတ်ခြင်းကို အထူးမြှင့်တင်ထားသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v23(img, n_cols):
    h, w = img.shape[:2]
    target_w = 1400 # Resolution ကို သင့်တင့်အောင် ချိန်ထားသည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # mag_ratio=1.5 နှင့် link_threshold=0.5 က ဂဏန်းများကို တစ်တွဲတည်းဖြစ်စေသည်
    results = reader.readtext(gray, paragraph=False, link_threshold=0.5, mag_ratio=1.5, min_size=5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

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

    # DYNAMIC COLUMN CALCULATION
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
            # တစ်တိုင်တည်းရှိ ဂဏန်းများကို ပေါင်းမည်
            combined_txt = "".join([i['text'] for i in bins[c]])
            
            # Ditto (။) စစ်ဆေးခြင်း
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1', '7', '/', 'I', '(', ')']) and len(combined_txt) <= 2
            
            if is_ditto:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: # ဂဏန်းတိုင်
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေးတိုင်
                        row_cells[c] = num
        final_grid.append(row_cells)

    # SMART FILL-DOWN (ထိုးကြေးအတွက်သာ အပေါ်ကကူးမည်)
    for c in range(n_cols):
        if c % 2 != 0: 
            last_val = ""
            for r in range(len(final_grid)):
                if final_grid[r][c] in ["DITTO", ""]:
                    if last_val: final_grid[r][c] = last_val
                else:
                    last_val = final_grid[r][c]
        else:
            # ဂဏန်းတိုင်တွင် DITTO ဖြစ်နေပါက အကွက်လွတ်ပြမည်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=500)
    
    if st.button(f"🔍 Scan {a_cols} Columns"):
        with st.spinner("အသေးစိတ် စစ်ဆေးနေပါသည်..."):
            res = process_v23(img, a_cols)
            st.session_state['data_v23'] = res

if 'data_v23' in st.session_state:
    edited = st.data_editor(st.session_state['data_v23'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets_v23(edited):
            st.success("Sheet ထဲသို့ ရောက်သွားပါပြီ!")
