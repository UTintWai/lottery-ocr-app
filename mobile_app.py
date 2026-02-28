import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery PC Pro v36", layout="wide")

@st.cache_resource
def load_ocr():
    # PC အတွက် Model ကို အဆင်သင့်ဖြစ်အောင် ကြိုတင် Load လုပ်ထားမည်
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v36(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        # --- TRUE GRID FORMATTING ---
        # အတန်းလိုက်မဖြစ်စေရန် ဒေတာကို nested list [ [], [], [] ] ပုံစံဖြင့် ပို့ရမည်
        final_batch = []
        for row in data:
            # ၈ တိုင်ထက်မပိုစေရန်နှင့် ဒေတာရှိမှယူရန်
            if any(str(c).strip() for c in row[:8]):
                # '062' ကဲ့သို့ ပေါ်ရန် ' ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                final_batch.append(clean_row)
        
        if final_batch:
            sheet.append_rows(final_batch, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Lottery PC Pro v36 (Strict 8-Column Grid)")

with st.sidebar:
    st.header("PC Mode Active")
    st.success("V36: Google Sheet ထဲသို့ Column A-H ဇယားအတိုင်း ကွက်တိဝင်စေရမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံကို ရွေးပါ (PC ဖြင့်သုံးရန် အကြံပြုသည်)", type=['jpg', 'jpeg', 'png'])

def process_v36(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1200 # Accuracy အတွက် Resolution မြှင့်ထားသည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR ဖတ်ခြင်း (PC RAM နိုင်သဖြင့် အကုန်တစ်ခါတည်းဖတ်မည်)
    results = reader.readtext(gray, paragraph=False)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text
        })

    if not raw_data: return []

    # ROW GROUPING
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 25:
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    # --- 8-COLUMN GRID ALIGNMENT ---
    final_table = []
    col_step = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_step)
            if 0 <= c_idx < 8:
                txt = re.sub(r'[^0-9"။=LVUYI/]', '', item['text'].upper())
                row_cells[c_idx] = (row_cells[c_idx] + txt).strip()
        
        # Formatting
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit():
                row_cells[c] = v.zfill(3)[:3]
            elif any(m in v for m in ['"', '။', '=', 'L', 'V', 'U', 'Y', 'I', '/']):
                row_cells[c] = "DITTO"
        final_table.append(row_cells)

    # Fill Down Amounts
    for c in [1, 3, 5, 7]:
        last = ""
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO" and last: final_table[r][c] = last
            elif final_table[r][c].isdigit(): last = final_table[r][c]
            
    # Clean Ditto from Number columns
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=600, caption="Uploaded Image")
    
    if st.button("🔍 Scan Data"):
        with st.spinner("PC စနစ်ဖြင့် အမြန်ဖတ်နေပါသည်..."):
            res = process_v36(img)
            st.session_state['data_v36'] = res

if 'data_v36' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (A ကနေ H အထိ)")
    # Edit လုပ်နိုင်သော ဇယားကို ပြမည်
    edited = st.data_editor(st.session_state['data_v36'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v36(edited):
            st.success("Sheet ထဲသို့ ဇယားကွက်အတိုင်း ကွက်တိသိမ်းဆည်းပြီးပါပြီ!")
