import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery Phone Pro v46", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v46(data):
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
                # Google Sheet ထဲမှာ 0 တွေမပျောက်အောင် ' ခံပေးခြင်း
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v46(img):
    reader = load_ocr()
    
    # --- 📸 IMAGE ENHANCEMENT (ဖုန်းပုံများအတွက် အထူးပြင်ဆင်ချက်) ---
    h, w = img.shape[:2]
    target_w = 1800 # Accuracy ပိုကောင်းရန် Resolution ကို အမြင့်ဆုံးမြှင့်သည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # Contrast မြှင့်ပြီး Noise ဖယ်ရှားခြင်း
    enhanced = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # OCR Scan
    results = reader.readtext(enhanced, paragraph=False)
    
    raw_data = []
    for (bbox, text, prob) in results:
        raw_data.append({
            'x': np.mean([p[0] for p in bbox]),
            'y': np.mean([p[1] for p in bbox]),
            'text': text
        })

    if not raw_data: return []

    # ROW GROUPING (စာကြောင်းခွဲခြင်း - ဖုန်းအတွက် Row Gap ကို ညှိထားသည်)
    raw_data.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - curr_row[-1]['y'] < 35: 
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
                txt = item['text'].strip()
                # Ditto သင်္ကေတများ (။ ၊ " =)
                if re.search(r'[။၊"=“”"„»«\-–—_]', txt):
                    row_cells[c_idx] = "DITTO"
                else:
                    # မှားတတ်သောစာလုံးများကို ပြင်ခြင်း (ဥပမာ S ကို 5 ပြောင်းခြင်း)
                    num = txt.upper().replace('S', '5').replace('G', '6').replace('I', '1').replace('B', '8')
                    num = re.sub(r'[^0-9]', '', num)
                    if num: row_cells[c_idx] = num
        
        # ၃ လုံးဂဏန်း Format (A, C, E, G တိုင်များအတွက်)
        for c in [0, 2, 4, 6]:
            if row_cells[c].isdigit():
                row_cells[c] = row_cells[c].zfill(3)[-3:]
        final_table.append(row_cells)

    # --- 🔥 VERTICAL DITTO FILL (အပေါ်ကထိုးကြေးကို အောက်ဖြည့်ခြင်း) ---
    for c in [1, 3, 5, 7]:
        active_amount = "" 
        for r in range(len(final_table)):
            val = str(final_table[r][c]).strip()
            if val.isdigit() and val != "":
                active_amount = val
            elif (val == "DITTO" or val == "") and active_amount != "":
                final_table[r][c] = active_amount

    # နံပါတ်တိုင်များရှိ Ditto များကို ရှင်းထုတ်ခြင်း
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 Lottery Smart Scanner v46")
st.info("ဖုန်းဖြင့်ရိုက်ထားသောပုံများကို ပိုမိုတိကျစွာဖတ်ပေးပြီး '။' သင်္ကေတများကို အလိုအလျောက် ဖြည့်ပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800, caption="Uploaded Image")
    
    if st.button("🔍 Scan with High Accuracy"):
        with st.spinner("ပုံရိပ်ကြည်လင်အောင်ပြုပြင်ပြီး အသေးစိတ်ဖတ်နေပါသည်..."):
            res = process_v46(img)
            st.session_state['data_v46'] = res

if 'data_v46' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    # Edit လုပ်နိုင်သော ဇယား
    edited = st.data_editor(st.session_state['data_v46'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v46(edited):
            st.success("Google Sheet ထဲသို့ ၈ တိုင်ကွက်တိ သိမ်းဆည်းပြီးပါပြီ!")
