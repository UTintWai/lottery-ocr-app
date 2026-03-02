import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery Pro v47", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v47(data):
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
                # '062' ကဲ့သို့ ပေါ်ရန် single quote ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v47(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1800 # Accuracy အတွက် အမြင့်ဆုံးထားသည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    
    # ပုံရိပ်ကို ကြည်လင်အောင် ပြုပြင်ခြင်း (Preprocessing)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)

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
        if raw_data[i]['y'] - curr_row[-1]['y'] < 32: 
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
                txt = item['text'].upper().strip()
                # Ditto သင်္ကေတများအား ဖမ်းယူခြင်း
                if re.search(r'[။၊"=“”"„»«\-–—_]', txt):
                    row_cells[c_idx] = "DITTO"
                else:
                    # မှားတတ်သောစာလုံးများကို Logic ဖြင့်ပြင်ခြင်း
                    txt = txt.replace('S','5').replace('G','6').replace('I','1').replace('B','8').replace('O','0')
                    num = re.sub(r'[^0-9]', '', txt)
                    if num: row_cells[c_idx] = num
        
        # နံပါတ်တိုင်များအတွက် ၃ လုံးဖြည့်ခြင်း
        for c in [0, 2, 4, 6]:
            if row_cells[c].isdigit():
                row_cells[c] = row_cells[c].zfill(3)[-3:]
        final_table.append(row_cells)

    # --- 🔥 VERTICAL FORCE-FILL (Ditto နှင့် အကွက်လွတ်များဖြည့်ခြင်း) ---
    for c in [1, 3, 5, 7]: # ထိုးကြေးတိုင်များ
        last_val = ""
        for r in range(len(final_table)):
            current = str(final_table[r][c]).strip()
            if current.isdigit() and current != "":
                last_val = current
            elif (current == "DITTO" or current == "") and last_val != "":
                final_table[r][c] = last_val

    # နံပါတ်တိုင်များရှိ Ditto များကို ရှင်းထုတ်ခြင်း
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 Lottery Phone Expert v47")
st.markdown("### ၈ တိုင်စလုံးကို အလိုအလျောက်ဖတ်ပြီး Ditto များကို အစားထိုးပေးမည်")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=700)
    
    if st.button("🔍 Scan with High Accuracy"):
        with st.spinner("AI က အမှားများကို ပြန်လည်စစ်ဆေးနေပါသည်..."):
            res = process_v47(img)
            st.session_state['data_v47'] = res

if 'data_v47' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    # Edit လုပ်နိုင်သော ဇယားကွက်
    edited = st.data_editor(st.session_state['data_v47'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💾 Save to Google Sheet"):
            if save_to_sheets_v47(edited):
                st.success("Google Sheet ထဲသို့ သိမ်းဆည်းပြီးပါပြီ!")
    with col2:
        if st.button("🗑️ Clear Data"):
            del st.session_state['data_v47']
            st.rerun()
