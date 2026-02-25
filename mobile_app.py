import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v10", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def process_v10(img, n_cols):
    h, w = img.shape[:2]
    # ပုံကို Resolution ပိုမြှင့်ပြီး Noise လျှော့ချပါမယ်
    img_resized = cv2.resize(img, (2000, int(h * (2000 / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR: paragraph=True က ဂဏန်းတွေကို ပေါင်းဖတ်ဖို့ ပိုအားကောင်းပါတယ်
    results = reader.readtext(gray, paragraph=True, x_ths=0.2, y_ths=0.1)
    
    raw_data = []
    for (bbox, text) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- ROW CLUSTERING ---
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 30 
    
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
    img_w = gray.shape[1]
    col_width = img_w / n_cols

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                txt = item['text']
                # Ditto Recognition: ပိုမိုတိကျသော သင်္ကေတများသာ ယူမည်
                is_ditto = any(m in txt for m in ['"', '။', '=', '||', 'LL', '`', 'V']) and len(txt) <= 2
                
                if is_ditto:
                    row_cells[c_idx] = "DITTO"
                else:
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: # ဂဏန်းတိုင် (၃ လုံး)
                            row_cells[c_idx] = num.zfill(3) if len(num) <= 3 else num[:3]
                        else: # ထိုးကြေးတိုင် (ဂဏန်းအပြည့်အစုံ)
                            row_cells[c_idx] = num
        final_grid.append(row_cells)

    # --- UPDATED FILL-DOWN LOGIC (ထိုးကြေးအတွက်သာ) ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်
            for r in range(1, len(final_grid)):
                # အကယ်၍ DITTO လို့ AI က သေချာဖတ်မိမှသာ အပေါ်ကဂဏန်းကို ကူးမည်
                # အကွက်လွတ်နေရင် မကူးတော့ဘဲ ဗောက်ချာအတိုင်း ထားမည်
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = final_grid[r-1][c]
                elif final_grid[r][c] == "":
                    # ဗောက်ချာမှာ အကွက်လွတ်ရင် အလွတ်ပဲထားပါမယ် (သို့မဟုတ် လိုအပ်လျှင် "0" ထားနိုင်သည်)
                    final_grid[r][c] = ""
        else: # ဂဏန်းတိုင်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

def save_to_sheets_v10(data):
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
st.title("🔢 Final Precision Scanner v10")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.markdown("""
    **V10 ပြင်ဆင်ချက်များ:**
    * ထိုးကြေးဂဏန်းများကို ပေါင်းဖတ်ရန် `paragraph` mode သုံးထားသည်။
    * အကွက်လွတ်တိုင်း အပေါ်ကဂဏန်း မကူးတော့ပါ။ `။` (Ditto) ပါမှသာ ကူးပါမည်။
    """)

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=450)
    
    if st.button("🔍 တိကျစွာ ဖတ်မည်"):
        with st.spinner("ဂဏန်းအပြည့်အစုံနှင့် Ditto Logic ကို စစ်ဆေးနေပါသည်..."):
            res = process_v10(img, a_cols)
            st.session_state['data_v10'] = res

if 'data_v10' in st.session_state:
    edited = st.data_editor(st.session_state['data_v10'], use_container_width=True)
    if st.button("💾 Google Sheet သို့ ပို့မည်"):
        if save_to_sheets_v10(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
