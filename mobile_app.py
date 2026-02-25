import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v13", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def process_v13(img, n_cols):
    h, w = img.shape[:2]
    target_w = 1500
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR: link_threshold ကို 0.1 အထိ လျှော့ချပြီး ဂဏန်းတွဲတွေကို အတင်းပေါင်းခိုင်းမယ်
    results = reader.readtext(gray, paragraph=False, link_threshold=0.1, mag_ratio=1.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- 1. ROW CLUSTERING (အတန်းခွဲခြင်း) ---
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 28 
    
    if raw_data:
        current_row = [raw_data[0]]
        for i in range(1, len(raw_data)):
            if raw_data[i]['y'] - current_row[-1]['y'] < y_threshold:
                current_row.append(raw_data[i])
            else:
                rows_list.append(current_row)
                current_row = [raw_data[i]]
        rows_list.append(current_row)

    # --- 2. DYNAMIC GRID ASSIGNMENT ---
    final_grid = []
    col_edges = np.linspace(0, target_w, n_cols + 1)

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        
        # Column တစ်ခုတည်းမှာရှိတဲ့ အပိုင်းအစတွေကို ပေါင်းရန်
        for item in row_items:
            # ဘယ် Column ထဲ ရောက်သလဲ စစ်ဆေးခြင်း
            c_idx = np.searchsorted(col_edges, item['x']) - 1
            if 0 <= c_idx < n_cols:
                txt = item['text']
                # Ditto Check
                is_ditto = any(m in txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1']) and len(txt) <= 2
                
                if is_ditto:
                    row_cells[c_idx] = "DITTO"
                else:
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: # ဂဏန်းတိုင် (၃ လုံး)
                            # ရှေ့က 0 ဖြည့်မယ် (003 ဖြစ်အောင်)
                            row_cells[c_idx] = num.zfill(3) if len(num) <= 3 else num[:3]
                        else: # ထိုးကြေးတိုင်
                            # အရင်ရှိပြီးသား ဂဏန်းနဲ့ ဆက်လိုက်မယ် (6 နဲ့ 0 တွေ့ရင် 60 ဖြစ်အောင်)
                            row_cells[c_idx] += num
        final_grid.append(row_cells)

    # --- 3. SMART AUTO-FILL (Ditto & Empty Amount) ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်များအတွက်သာ
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                # အကွက်လွတ်နေရင် သို့မဟုတ် DITTO ဖြစ်နေရင် အပေါ်ကဂဏန်းယူမယ်
                if val == "" or val == "DITTO":
                    if last_amt != "":
                        final_grid[r][c] = last_amt
                else:
                    # ဂဏန်းအသစ်တွေ့ရင် အဲ့ဒါကိုပဲယူပြီး နောက်အကွက်အတွက် မှတ်ထားမယ်
                    last_amt = val
        else: # ဂဏန်းတိုင်များအတွက်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

# (Google Sheets function များ အရင်အတိုင်းပဲမို့ ချန်လှပ်ထားပါမည်)

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
                res = process_v11(img, a_cols) # type: ignore
                st.session_state['data_v11'] = res
            except Exception as e:
                st.error("Memory ပြည့်သွားပါပြီ။ ကျေးဇူးပြု၍ ပုံအရွယ်အစား လျှော့ပြီး ပြန်တင်ပေးပါ။")

if 'data_v11' in st.session_state:
    edited = st.data_editor(st.session_state['data_v11'], use_container_width=True)
    if st.button("💾 Google Sheet သို့ ပို့မည်"):
        if save_to_sheets(edited): # type: ignore
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
