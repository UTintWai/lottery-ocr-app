import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v18", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.title("🔢 Lottery Pro Scanner (v18)")

with st.sidebar:
    # တိုင်အရေအတွက် ရွေးချယ်မှု
    a_cols = st.selectbox("တိုင်အရေအတွက် ရွေးပါ", [6, 8], index=1)
    st.warning("၈ တိုင်ဗောက်ချာဆိုလျှင် ပုံကို တည့်တည့်နှင့် အနားသတ်များ ပါအောင် ရိုက်ပေးပါဗျ။")

# --- FILE UPLOADER ---
up_file = st.file_uploader("ဗောက်ချာပုံ ရွေးရန်", type=['jpg', 'jpeg', 'png'])

def process_v18(img, n_cols):
    h, w = img.shape[:2]
    # ၈ တိုင်အတွက် resolution ပိုလိုအပ်လို့ 1600px အထိ မြှင့်ပါမယ်
    target_w = 1600
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ၈ တိုင်အတွက် စာလုံးသေးများကို ဖတ်နိုင်ရန် mag_ratio မြှင့်ထားသည်
    results = reader.readtext(gray, paragraph=False, link_threshold=0.1, mag_ratio=1.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # ROW CLUSTERING
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 30 # အတန်းအကွာအဝေး
    
    current_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - current_row[-1]['y'] < y_threshold:
            current_row.append(raw_data[i])
        else:
            rows_list.append(current_row)
            current_row = [raw_data[i]]
    rows_list.append(current_row)

    # PRECISION GRID ASSIGNMENT
    final_grid = []
    col_edges = np.linspace(0, target_w, n_cols + 1)

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        bins = [[] for _ in range(n_cols)]
        
        for item in row_items:
            # item တစ်ခုစီကို ဘယ် Column ထဲမှာ ရှိသလဲ ရှာဖွေခြင်း
            for c in range(n_cols):
                if col_edges[c] <= item['x'] < col_edges[c+1]:
                    bins[c].append(item)
                    break
        
        for c in range(n_cols):
            bins[c].sort(key=lambda k: k['x'])
            combined_txt = "".join([i['text'] for i in bins[c]])
            
            # Ditto Logic
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1', '7']) and len(combined_txt) <= 2
            
            if is_ditto:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: # ၃ လုံးဂဏန်းတိုင်
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေးတိုင်
                        row_cells[c] = num
        final_grid.append(row_cells)

    # SMART FILL-DOWN (ထိုးကြေးအတွက်သာ)
    for c in range(n_cols):
        if c % 2 != 0: 
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                if val == "DITTO" or val == "":
                    if last_amt != "":
                        final_grid[r][c] = last_amt
                else:
                    last_amt = val
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

# --- MAIN LOGIC ---
if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=500, caption="ဗောက်ချာပုံ")
    
    if st.button("🔍 ၈ တိုင် Scan လုပ်မည်"):
        with st.spinner("အချက်အလက်များကို တွက်ချက်နေပါသည်..."):
            res = process_v18(img, a_cols)
            if res:
                st.session_state['data_v18'] = res
            else:
                st.error("ဒေတာ မဖတ်မိပါဗျ။ ပုံကို ပြန်စစ်ပေးပါ။")

if 'data_v18' in st.session_state:
    st.write("ရလဒ်များကို စစ်ဆေးပါ-")
    edited = st.data_editor(st.session_state['data_v18'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets(edited):
            st.success("အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
