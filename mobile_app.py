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
    # PC RAM ကို အသုံးချပြီး OCR Model ကို ကြိုတင် Load လုပ်ထားခြင်း
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v43(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        # PC မှာ Local သုံးနေလျှင် secrets အစား "creds.json" ဖိုင်လမ်းကြောင်း ပြင်သုံးနိုင်သည်
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
            # append_rows က Nested List ကို ဇယားကွက်အတိုင်း ခွဲသွင်းပေးပါသည်
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v43(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600 # Resolution မြှင့်ပြီး Ditto တွေကို ပိုဖမ်းမိအောင်လုပ်ခြင်း
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
                # "။" သို့မဟုတ် တူညီသည့် သင်္ကေတများကို DITTO အဖြစ် သတ်မှတ်ခြင်း
                is_ditto = bool(re.search(r'[။၊"=“”"„»«\-–—_]', orig_text))
                
                if is_ditto:
                    row_cells[c_idx] = "DITTO"
                else:
                    num_only = re.sub(r'[^0-9]', '', orig_text)
                    if num_only:
                        row_cells[c_idx] = num_only
        
        # Numbers များကို ၃ လုံးဖြည့်ခြင်း
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit() and v != "":
                row_cells[c] = v.zfill(3)[:3]
        
        final_table.append(row_cells)

    # --- SMART FILL DOWN (DITTO Logic) ---
    for c in [1, 3, 5, 7]: # Amount တိုင်များအတွက်
        last_val = ""
        for r in range(len(final_table)):
            val = final_table[r][c]
            if val == "DITTO" and last_val:
                final_table[r][c] = last_val
            elif val.isdigit() and val != "":
                last_val = val
            elif val == "DITTO" and not last_val:
                final_table[r][c] = "" 

    # Number တိုင်များရှိ DITTO များကို ရှင်းထုတ်ခြင်း
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO":
                final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 PC Precise Scanner v43 (Choice A)")
st.info("ဗောက်ချာမှ '။' သင်္ကေတများကို DITTO အဖြစ် ဖမ်းယူပြီး အလိုအလျောက် ဂဏန်းဖြည့်ပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800, caption="မူရင်းပုံ")
    
    if st.button("🔍 Scan All 8 Columns"):
        with st.spinner("AI က ၈ တိုင်စလုံးရှိ Ditto များကို ရှာဖွေနေပါသည်..."):
            res = process_v43(img)
            st.session_state['data_v43'] = res

if 'data_v43' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (Column A မှ H)")
    edited = st.data_editor(st.session_state['data_v43'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v43(edited):
            st.success("Google Sheet ထဲသို့ ဇယားကွက်အတိုင်း အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
