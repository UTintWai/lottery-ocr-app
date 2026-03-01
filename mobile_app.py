import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery PC Pro v44", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v44(data):
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
                # Sheet ထဲမှာ 0 တွေမပျောက်အောင် ' ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v44(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600 
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR Scan (Ditto များကို ပိုမိစေရန်)
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

    # --- 8-COLUMN GRID MAPPING ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                orig_text = item['text'].strip()
                
                # Ditto သင်္ကေတများအားလုံးကို ရှာဖွေခြင်း (။ ၊ ၊ " = - _ စသည်ဖြင့်)
                is_ditto = bool(re.search(r'[။၊"=“”"„»«\-–—_]', orig_text))
                
                if is_ditto:
                    row_cells[c_idx] = "DITTO"
                else:
                    # ဂဏန်းသီးသန့် (0-9) ယူမည်
                    num_only = re.sub(r'[^0-9]', '', orig_text)
                    if num_only:
                        row_cells[c_idx] = num_only
        
        # ၃ လုံးဂဏန်း formatting
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit() and v != "":
                row_cells[c] = v.zfill(3)[:3]
        
        final_table.append(row_cells)

    # --- 🔥 DITTO AUTO-CLONE LOGIC (ဤအပိုင်းက ထိုးကြေးကို ကူးပေးပါသည်) ---
    for c in [1, 3, 5, 7]: # ထိုးကြေးတိုင်များ (Amount Columns)
        current_amount = "" # အပေါ်ဆုံးမှစ၍ မှတ်ထားမည်
        for r in range(len(final_table)):
            val = str(final_table[r][c]).strip()
            
            if val.isdigit() and val != "":
                # ဂဏန်းတွေ့ရင် နောက်ထပ် Ditto တွေအတွက် ဒီဂဏန်းကို မှတ်ထားမည်
                current_amount = val
            elif val == "DITTO":
                # Ditto တွေ့ရင် အပေါ်က မှတ်ထားတဲ့ ဂဏန်းကို အစားထိုးမည်
                final_table[r][c] = current_amount
            elif val == "" and current_amount != "":
                # တစ်ခါတလေ AI က Ditto ကို မဖတ်နိုင်ဘဲ ကွက်လပ်ဖြစ်နေရင်လည်း အပေါ်ကဂဏန်းကို ဖြည့်ပေးမည်
                final_table[r][c] = current_amount

    # နံပါတ်တိုင်များရှိ Ditto များကို ဖျက်ထုတ်ခြင်း (နံပါတ်က Ditto မရှိနိုင်ပါ)
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO":
                final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 PC Ultimate Scanner v44 (Ditto Cloning Fix)")
st.info("ဗောက်ချာမှ '။' သင်္ကေတများကို အပေါ်မှ ထိုးကြေးဂဏန်းများအတိုင်း အလိုအလျောက် ကူးယူဖြည့်သွင်းပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 Scan and Auto-Fill Ditto"):
        with st.spinner("AI က ထိုးကြေးများကို တွက်ချက်ကူးယူနေပါသည်..."):
            res = process_v44(img)
            st.session_state['data_v44'] = res

if 'data_v44' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    # Edit လုပ်နိုင်သော ဇယား
    edited = st.data_editor(st.session_state['data_v44'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v44(edited):
            st.success("Google Sheet ထဲသို့ ၈ တိုင်ကွက်တိ သိမ်းဆည်းပြီးပါပြီ!")
