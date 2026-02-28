import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery PC Pro v38", layout="wide")

@st.cache_resource
def load_ocr():
    # PC RAM အတွက် အဆင်သင့်ဖြစ်အောင် model ကြိုတင် load လုပ်ထားမည်
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v38(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        # --- TRUE GRID MAPPING ---
        # Google Sheet ထဲကို ဇယားကွက်အတိုင်းဝင်ဖို့ [ [], [], [] ] nested list ပုံစံပြင်ခြင်း
        upload_batch = []
        for row in data:
            if any(str(c).strip() for c in row[:8]):
                # '062' ကဲ့သို့ ပေါ်ရန် single quote ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                upload_batch.append(clean_row)
        
        if upload_batch:
            # append_rows သည် nested list ကို ဇယားကွက်အတိုင်း ခွဲထည့်ပေးပါသည်
            sheet.append_rows(upload_batch, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

st.title("🔢 Lottery PC Ultimate v38 (Table Grid Fix)")

with st.sidebar:
    st.header("PC Mode Active")
    st.info("Version 38: PC ဖြင့်ဖတ်ပြီး Google Sheet ထဲသို့ ၈ တိုင် (Column A-H) ဇယားအတိုင်း ကွက်တိဝင်စေမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံကို ရွေးပါ", type=['jpg', 'jpeg', 'png'])

def process_v38(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    # PC အတွက် Resolution ကို အမြင့်ဆုံး (1400px) ထားပြီး ဖတ်မည်
    target_w = 1400 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # PC အတွက် အမြန်နှုန်းမြှင့်တင်ရန် တစ်ခါတည်းဖတ်မည်
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
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 25:
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    # --- 8-COLUMN STRICT MAPPING ---
    final_table = []
    col_step = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_step)
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

    # Smart Fill Down (ထိုးကြေးတိုင်များအတွက်)
    for c in [1, 3, 5, 7]:
        last = ""
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO" and last: final_table[r][c] = last
            elif final_table[r][c].isdigit(): last = final_table[r][c]
            
    # Clean Numbers columns (DITTO စာသားများဖယ်ရန်)
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=700, caption="Uploaded Image")
    
    if st.button("🔍 Scan Data"):
        with st.spinner("PC စနစ်ဖြင့် အမြန်ဖတ်နေပါသည်..."):
            res = process_v38(img)
            st.session_state['data_v38'] = res

if 'data_v3_8' in st.session_state or 'data_v38' in st.session_state:
    data_to_show = st.session_state.get('data_v38', [])
    st.subheader("စစ်ဆေးရန် ဇယား (A မှ H တိုင်)")
    edited = st.data_editor(data_to_show, use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v38(edited):
            st.success("Google Sheet ထဲသို့ ဇယားကွက်အတိုင်း ကွက်တိသိမ်းဆည်းပြီးပါပြီ!")
