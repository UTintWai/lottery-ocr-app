import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v22", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- GOOGLE SHEETS FUNCTION (အမှားရှာရလွယ်အောင် ပြင်ထားသည်) ---
def save_to_sheets_v22(data):
    try:
        # 1. Connection တည်ဆောက်ခြင်း
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        if "gcp_service_account" not in st.secrets:
            st.error("Secret setting ထဲမှာ gcp_service_account မတွေ့ပါဗျ။")
            return False
            
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # 2. Sheet ဖွင့်ခြင်း (နာမည်ကို သေချာစစ်ပါ)
        sheet_name = "LotteryData" 
        try:
            sh = client.open(sheet_name)
            sheet = sh.get_worksheet(0) # ပထမဆုံး tab ကို ယူမည်
        except gspread.SpreadsheetNotFound:
            st.error(f"Google Sheet အမည် '{sheet_name}' ကို ရှာမတွေ့ပါဗျ။ နာမည်မှန်အောင် ပြန်ပေးပါ။")
            return False

        # 3. ဒေတာပြင်ဆင်ခြင်း (အကွက်လွတ်များ ဖယ်ထုတ်ခြင်း)
        formatted = []
        for row in data:
            if any(str(cell).strip() != "" for cell in row):
                # Google Sheet မှာ 003 လို့ ပေါ်ဖို့ ရှေ့က ' ခံပေးရပါမယ်
                formatted_row = [f"'{str(cell)}" if str(cell).strip() != "" else "" for cell in row]
                formatted.append(formatted_row)
        
        if formatted:
            sheet.append_rows(formatted)
            return True
        else:
            st.warning("သိမ်းစရာ ဒေတာ မရှိပါဗျ။")
            return False
            
    except Exception as e:
        st.error(f"Sheet ထဲပို့ရာတွင် အခက်အခဲရှိနေပါသည်: {str(e)}")
        return False

# --- UI & SCANNING LOGIC ---
st.title("🔢 Lottery Precision Scanner v22")

with st.sidebar:
    a_cols = st.selectbox("ဗောက်ချာပါ တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    if st.button("Reset Data"):
        if 'data_v22' in st.session_state:
            del st.session_state['data_v22']
            st.rerun()

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v22(img, n_cols):
    h, w = img.shape[:2]
    target_w = 1200
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    results = reader.readtext(gray, paragraph=False, link_threshold=0.3, mag_ratio=1.1)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 25
    if raw_data:
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
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1', '7', '/', 'I']) and len(combined_txt) <= 2
            
            if is_ditto:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: row_cells[c] = num
        final_grid.append(row_cells)

    # Fill Down Logic
    for c in range(n_cols):
        if c % 2 != 0:
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                if val in ["DITTO", ""]:
                    if last_amt: final_grid[r][c] = last_amt
                else: last_amt = val
        else:
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
    return final_grid

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=450)
    
    if st.button(f"🔍 Scan {a_cols} Columns"):
        with st.spinner("AI ဖတ်နေပါသည်..."):
            res = process_v22(img, a_cols)
            st.session_state['data_v22'] = res

if 'data_v22' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    edited_data = st.data_editor(st.session_state['data_v22'], use_container_width=True)
    
    if st.button("💾 Google Sheet သို့ ပို့မည်"):
        with st.spinner("Sheet ထဲသို့ ပို့နေပါသည်..."):
            if save_to_sheets_v22(edited_data):
                st.success("အောင်မြင်စွာ ပို့ပြီးပါပြီ! Google Sheet ကို သွားစစ်ကြည့်ပါဗျ။")
                # ပို့ပြီးရင် data ကို clear လုပ်ချင်ရင်:
                # del st.session_state['data_v22']
