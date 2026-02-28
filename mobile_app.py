import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery PC Pro v40", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v40(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        # PC Local သုံးနေလျှင် secrets အစား "creds.json" ဟု ပြင်သုံးနိုင်သည်
        creds_dict = st.secrets["gcp_service_account"] 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        # --- BATCH GRID PROCESSING ---
        # အတန်းလိုက်မဖြစ်စေရန် list ထဲမှာ list ပြန်ထည့်တဲ့ [ [Row1], [Row2] ] format ပြောင်းခြင်း
        table_rows = []
        for row in data:
            if any(str(c).strip() for c in row[:8]):
                # Google Sheet မှ 0 တွေကိုစာသားအဖြစ်သိမ်းရန် ' ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            # append_rows သည် nested list ကို ဇယားကွက်အတိုင်း ခွဲသွင်းပေးပါသည်
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v40(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1400 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR Scan
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
        if raw_data[i]['y'] - curr_row[-1]['y'] < 28: # စာကြောင်းအမြင့်ညှိခြင်း
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    # --- 8-COLUMN TABLE MAPPING ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            # X coordinate ကိုကြည့်ပြီး ဘယ် Column ထဲရောက်ရမလဲ တွက်ခြင်း
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                txt = re.sub(r'[^0-9"။=LVUYI/]', '', item['text'].upper())
                row_cells[c_idx] = (row_cells[c_idx] + txt).strip()
        
        # Format Data (3-digit Numbers & Ditto)
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit():
                row_cells[c] = v.zfill(3)[:3]
            elif any(m in v for m in ['"', '။', '=', 'L', 'V', 'U', 'Y', 'I', '/']):
                row_cells[c] = "DITTO"
        final_table.append(row_cells)

    # Smart Fill Down (Amounts)
    for c in [1, 3, 5, 7]:
        last_amt = ""
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO" and last_amt: final_table[r][c] = last_amt
            elif final_table[r][c].isdigit(): last_amt = final_table[r][c]
            
    # Number Columns Cleaning
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 PC Ultimate 8-Column Scanner v40")
st.write("PC အတွက် ဇယားကွက်အမှန်ထွက်စေမည့် Version ဖြစ်ပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 Scan All 8 Columns"):
        with st.spinner("PC Engine က ဇယားကွက်အတိုင်း ဖတ်နေပါသည်..."):
            res = process_v40(img)
            st.session_state['data_v40'] = res

if 'data_v40' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (A မှ H)")
    # Data Editor တွင် ဇယားကွက် ၈ ခု ပေါ်ရမည်
    edited = st.data_editor(st.session_state['data_v40'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v40(edited):
            st.success("Google Sheet ထဲသို့ ဇယားကွက်အတိုင်း ကွက်တိ ရောက်သွားပါပြီ!")
