import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v11", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM ချွေတာရန် model ကို ပေါ့ပေါ့ပါးပါးပဲ load လုပ်ပါမယ်
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models')

reader = load_ocr()

def process_v11(img, n_cols):
    # 1. Image Scaling (RAM Crash မဖြစ်စေရန် 1200px - 1500px ကြားပဲ ထားပါမယ်)
    h, w = img.shape[:2]
    target_w = 1300
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 2. OCR (link_threshold ကို သုံးပြီး ဂဏန်းတွဲတွေကို ပေါင်းဖတ်ခိုင်းမယ်)
    # paragraph=False ပြန်ထားပြီး link_threshold နဲ့ ထိုးကြေးကို ဖမ်းပါမယ်
    results = reader.readtext(gray, paragraph=False, link_threshold=0.3, add_margin=0.1)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- ROW CLUSTERING ---
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

    # --- GRID CALCULATION ---
    final_grid = []
    col_width = target_w / n_cols

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        
        # တစ်တိုင်တည်းမှာ စာလုံးကွဲနေရင် ပေါင်းပေးဖို့ temp list
        temp_bins = [[] for _ in range(n_cols)]
        
        for item in row_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                temp_bins[c_idx].append(item)
        
        for c in range(n_cols):
            # Column အလိုက် စာသားများကို x-position အလိုက်စီပြီး ပေါင်းမယ် (ဥပမာ 6 နဲ့ 0 ပေါင်းဖို့)
            temp_bins[c].sort(key=lambda k: k['x'])
            combined_txt = "".join([i['text'] for i in temp_bins[c]])
            
            # Ditto Logic
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V']) and len(combined_txt) <= 2
            
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

    # --- DITTO FILL-DOWN (ထိုးကြေးအတွက်သာ) ---
    for c in range(n_cols):
        if c % 2 != 0:
            last_amt = ""
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = last_amt
                elif final_grid[r][c] != "":
                    last_amt = final_grid[r][c]
        else:
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
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
st.title("🔢 Lottery Scanner v11 (RAM Safe)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.info("RAM Error မတက်စေရန်နှင့် ဂဏန်းများမကျန်စေရန် Logic ကို ပြန်လည်ညှိနှိုင်းထားပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button("🔍 Scan လုပ်မယ်"):
        with st.spinner("အချက်အလက်များကို စစ်ဆေးနေပါသည်..."):
            try:
                res = process_v11(img, a_cols)
                st.session_state['data_v11'] = res
            except Exception as e:
                st.error("Memory ပြည့်သွားပါပြီ။ ကျေးဇူးပြု၍ ပုံအရွယ်အစား လျှော့ပြီး ပြန်တင်ပေးပါ။")

if 'data_v11' in st.session_state:
    edited = st.data_editor(st.session_state['data_v11'], use_container_width=True)
    if st.button("💾 Google Sheet သို့ ပို့မည်"):
        if save_to_sheets(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
