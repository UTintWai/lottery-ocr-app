import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery PC Ditto-Fix v43", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v43(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"] 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        table_rows = []
        for row in data:
            if any(str(c).strip() for c in row[:8]):
                # Sheet ထဲမှာ 0 တွေမပျောက်အောင် ' ခံပြီးသွင်းမည်
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v43(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # ပုံကို အလင်းအမှောင်ညှိပြီး ဖတ်ခြင်း (Ditto ပိုမိစေရန်)
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
        if raw_data[i]['y'] - curr_row[-1]['y'] < 28:
            curr_row.append(raw_data[i])
        else:
            rows.append(curr_row)
            curr_row = [raw_data[i]]
    rows.append(curr_row)

    # --- 8-COLUMN GRID MAPPING WITH DITTO FOCUS ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                orig_text = item['text'].strip()
                # ဂဏန်းမဟုတ်တဲ့ "။" ၊ "၊" ၊ "=" ၊ "\"" သင်္ကေတတွေကို ရှာဖွေခြင်း
                is_ditto = bool(re.search(r'[။၊"=“”"„»«\-–—_]', orig_text))
                
                if is_ditto:
                    row_cells[c_idx] = "DITTO"
                else:
                    # ဂဏန်းသီးသန့်ယူခြင်း
                    num_only = re.sub(r'[^0-9]', '', orig_text)
                    if num_only:
                        row_cells[c_idx] = num_only
        
        # Formatting (3-digits for numbers)
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit() and v != "":
                row_cells[c] = v.zfill(3)[:3]
        
        final_table.append(row_cells)

    # --- SMART FILL DOWN (DITTO Logic) ---
    # Amount တိုင်တွေမှာ DITTO တွေ့ရင် အပေါ်ကဂဏန်းကို ယူထည့်မည်
    for c in [1, 3, 5, 7]:
        last_val = ""
        for r in range(len(final_table)):
            val = final_table[r][c]
            if val == "DITTO" and last_val:
                final_table[r][c] = last_val
            elif val.isdigit() and val != "":
                last_val = val
            elif val == "DITTO" and not last_val:
                final_table[r][c] = "" # အပေါ်မှာ ဘာမှမရှိရင် အလွတ်ထားမည်

    # Number တိုင်တွေမှာ DITTO တွေ့ရင် အလွတ်ပေးမည် (ဂဏန်းက Ditto မရှိနိုင်သဖြင့်)
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO":
                final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 PC Precise Scanner v43 (Ditto Fixed)")
st.info("ဗောက်ချာမှ '။' သင်္ကေတများကို အလိုအလျောက် DITTO အဖြစ်ပြောင်းလဲပြီး အပေါ်ကထိုးကြေးများကို ဖြည့်ပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 Scan All 8 Columns"):
        with st.spinner("Ditto သင်္ကေတများကို ရှာဖွေဖတ်နေပါသည်..."):
            res = process_v43(img)
            st.session_state['data_v43'] = res

if 'data_v43' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (A မှ H)")
    # Edit လုပ်နိုင်သော ဇယားကိုပြမည်
    edited = st.data_editor(st.session_state['data_v43'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v43(edited):
            st.success("Google Sheet ထဲသို့ ဇယားကွက်အတိုင်း အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
