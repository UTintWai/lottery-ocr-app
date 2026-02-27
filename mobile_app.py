import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v26", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM crash မဖြစ်အောင် model storage ကို ပိုထိန်းထားပါတယ်
    return easyocr.Reader(['en'], gpu=False)

try:
    reader = load_ocr()
except Exception:
    st.error("OCR Model တက်လာဖို့ ခေတ္တစောင့်ပေးပါ သို့မဟုတ် Refresh လုပ်ပေးပါဗျ။")

def save_to_sheets_v26(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").get_worksheet(0)
        
        # Google Sheet မှာ 062 ကဲ့သို့ ပေါ်ရန် ' ခံပေးမည်
        formatted = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Lottery Precision Scanner v26")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    st.success("V26: Memory Management နှင့် လက်ရေး Ditto Logic ကို အထူးမြှင့်တင်ထားသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v26(img, n_cols):
    h, w = img.shape[:2]
    # RAM ချွေတာရန် resolution ကို ၁၀၀၀ သာ ထားပါမည်
    target_w = 1000
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR Settings (လက်ရေးအတွက် link_threshold ကို ချိန်ညှိထားသည်)
    results = reader.readtext(gray, paragraph=False, link_threshold=0.3, mag_ratio=1.0)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # ROW CLUSTERING (စာကြောင်းများ မလွတ်စေရန် threshold ကို ၂၅ ထားသည်)
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 25
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
            txt = "".join([i['text'] for i in bins[c]])
            
            # --- လက်ရေး Ditto (။) စစ်ဆေးခြင်း ---
            # လက်ရေး "။" ကို AI မှ 4, u, v, n, 11, / စသည်ဖြင့် မှားဖတ်လေ့ရှိသည်ကို စစ်ဆေးမည်
            is_ditto_char = any(m in txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '/', '11', 'I', '(', ')', 'N'])
            
            if is_ditto_char and len(re.sub(r'[^0-9]', '', txt)) < 3:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', txt)
                if num:
                    if c % 2 == 0: # ဂဏန်းတိုင်
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေးတိုင်
                        row_cells[c] = num
        final_grid.append(row_cells)

    # --- Smart Fill Down Logic ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်အတွက်သာ အပေါ်ကကူးမည်
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                if val in ["DITTO", ""] and last_amt:
                    final_grid[r][c] = last_amt
                elif val not in ["DITTO", ""]:
                    last_amt = val
        else: # ဂဏန်းတိုင်တွင် Ditto ဖြစ်နေပါက ဖျောက်မည်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=450)
    
    if st.button(f"🔍 Scan {a_cols} Columns"):
        with st.spinner("ဖတ်နေပါသည်..."):
            try:
                res = process_v26(img, a_cols)
                st.session_state['data_v26'] = res
            except Exception as e:
                st.error("Memory လောက်အောင် ပုံကို အရွယ်အစား လျှော့တင်ပေးပါဗျ။")

if 'data_v26' in st.session_state:
    edited = st.data_editor(st.session_state['data_v26'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets_v26(edited):
            st.success("Sheet ထဲသို့ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
