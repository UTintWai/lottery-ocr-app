import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v35", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v35(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        # Sheet အမည်ကို သေချာစစ်ပါ
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        # --- BATCH SAVING (ဇယားကွက်အတိုင်း ပို့ခြင်း) ---
        all_rows = []
        for row in data:
            if any(str(c).strip() for c in row[:8]): # ဒေတာရှိတဲ့ အတန်းပဲယူမယ်
                formatted_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                all_rows.append(formatted_row)
        
        if all_rows:
            sheet.append_rows(all_rows) # append_rows က ဇယားအတိုင်း အပုံလိုက်သွင်းပေးတာပါ
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Precise Table Scanner v35")

with st.sidebar:
    st.header("Settings")
    st.info("Version 35: Google Sheet ထဲသို့ ဇယားကွက်အတိုင်း (Column A-H) ကွက်တိ သွင်းပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံ တင်ပေးပါ (PC သို့မဟုတ် ဖုန်း)", type=['jpg', 'jpeg', 'png'])

def process_v35(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1200 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # PC အတွက် ပိုမြန်အောင် paragraph mode သုံးမည်
    results = reader.readtext(gray, paragraph=False)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text
        })

    if not raw_data: return []

    # ROW GROUPING (စာကြောင်းခွဲခြင်း)
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    if raw_data:
        curr_row = [raw_data[0]]
        for i in range(1, len(raw_data)):
            if raw_data[i]['y'] - curr_row[-1]['y'] < 25:
                curr_row.append(raw_data[i])
            else:
                rows.append(curr_row)
                curr_row = [raw_data[i]]
        rows.append(curr_row)

    # --- 8-COLUMN GRID MAPPING ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                txt = re.sub(r'[^0-9"။=LVUYI/]', '', item['text'].upper())
                row_cells[c_idx] = (row_cells[c_idx] + txt).strip()
        
        # Formatting (၃ လုံးဂဏန်း နှင့် Ditto)
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit():
                row_cells[c] = v.zfill(3)[:3]
            elif any(m in v for m in ['"', '။', '=', 'L', 'V', 'U', 'Y', 'I', '/']):
                row_cells[c] = "DITTO"
        final_table.append(row_cells)

    # Auto-Fill Amounts
    for c in [1, 3, 5, 7]:
        last = ""
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO" and last: final_table[r][c] = last
            elif final_table[r][c].isdigit(): last = final_table[r][c]
            
    # Clean Numbers Column
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=600, caption="ဗောက်ချာမူရင်း")
    
    if st.button("🔍 Scan ဖတ်မည်"):
        with st.spinner("PC စနစ်ဖြင့် အမြန်ဖတ်နေပါသည်..."):
            res = process_v35(img)
            st.session_state['data_v35'] = res

if 'data_v35' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    # ဇယားကို တိုင် ၈ တိုင် (A to H) အတိုင်း ပြပေးမည်
    edited = st.data_editor(st.session_state['data_v35'], 
                            column_config={str(i): f"Col {i+1}" for i in range(8)},
                            use_container_width=True)
    
    if st.button("💾 Google Sheet ထဲသို့ ဇယားအတိုင်းသွင်းမည်"):
        if save_to_sheets_v35(edited):
            st.success("Google Sheet ထဲသို့ Column A မှ H အထိ ဇယားကွက်အတိုင်း ရောက်သွားပါပြီ!")
