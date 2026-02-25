import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v12", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def process_v12(img, n_cols):
    h, w = img.shape[:2]
    # RAM Crash မဖြစ်စေရန်နှင့် တိကျစေရန် 1400px ထားပါမယ်
    target_w = 1400
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR results: mag_ratio မြှင့်ထားခြင်းဖြင့် စာလုံးသေးလေးတွေကို ပိုဖတ်နိုင်စေပါတယ်
    results = reader.readtext(gray, paragraph=False, mag_ratio=1.2, link_threshold=0.2)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- ROW CLUSTERING ---
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

    # --- GRID CALCULATION ---
    final_grid = []
    col_width = target_w / n_cols

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        
        # အကွက်တစ်ခုတည်းမှာရှိတဲ့ ဂဏန်းအပိုင်းအစတွေကို ပေါင်းဖို့ temp list
        temp_bins = [[] for _ in range(n_cols)]
        for item in row_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                temp_bins[c_idx].append(item)
        
        for c in range(n_cols):
            temp_bins[c].sort(key=lambda k: k['x'])
            # တစ်အိမ်တည်းက စာလုံးတွေကို ဆက်လိုက်ခြင်း
            combined_txt = "".join([i['text'] for i in temp_bins[c]])
            
            # Ditto Detection
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1']) and len(combined_txt) <= 2
            
            if is_ditto:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: # ၃ လုံးဂဏန်း (Column 0, 2, 4, 6)
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေး (Column 1, 3, 5, 7)
                        row_cells[c] = num
        final_grid.append(row_cells)

    # --- 💡 SMART FILL LOGIC (။ မပါရင်တောင် ဖြည့်မည့်စနစ်) ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်အတွက်သာ
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                # အကွက်လွတ်နေရင် သို့မဟုတ် DITTO ဖြစ်နေရင် အပေါ်ကဂဏန်းယူမယ်
                if val == "" or val == "DITTO":
                    if last_amt != "":
                        final_grid[r][c] = last_amt
                else:
                    # တကယ်လို့ ဂဏန်းအသစ် (ဥပမာ 60) တွေ့ရင် အဲ့ဒါကိုပဲ ယူပြီး မှတ်ထားမယ်
                    last_amt = val
        else: # ၃ လုံးဂဏန်းတိုင်
            for r in range(len(final_grid)):
                # ဂဏန်းတိုင်မှာ Ditto တွေ့ရင် အပေါ်ကမကူးဘဲ ဖျက်လိုက်မယ်
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

# (save_to_sheets UI အပိုင်းက အရင်အတိုင်းပဲမို့ ချန်လှပ်ထားပါမယ်...)

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
