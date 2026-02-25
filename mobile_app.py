import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Ultimate Scanner v7", layout="wide")

@st.cache_resource
def load_ocr():
    # recognition model ကို ပိုမိုတိကျစေရန် paragraph=True သုံးဖို့အတွက် logic ပြင်မယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def process_v7(img, n_cols):
    h, w = img.shape[:2]
    # ပုံကို Resolution အားကောင်းအောင် ထိန်းညှိခြင်း
    img_resized = cv2.resize(img, (1600, int(h * (1600 / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # OCR results (paragraph=True က စာလုံးတွဲတွေကို ပိုမှန်အောင် ဖတ်ပေးနိုင်ပါတယ်)
    results = reader.readtext(gray, paragraph=False, width_ths=0.5)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- ADVANCED ROW CLUSTERING ---
    # y-coordinate အလိုက် အုပ်စုခွဲပြီး အတန်းဖော်ထုတ်ခြင်း
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    if raw_data:
        current_row = [raw_data[0]]
        y_threshold = 25 # အတန်းအကွာအဝေး ညှိရန်

        for i in range(1, len(raw_data)):
            if raw_data[i]['y'] - current_row[-1]['y'] < y_threshold:
                current_row.append(raw_data[i])
            else:
                rows_list.append(current_row)
                current_row = [raw_data[i]]
        rows_list.append(current_row)

    # --- PRECISE GRID ASSIGNMENT ---
    final_grid = []
    img_w = gray.shape[1]
    # Column များကို x-coordinate အလိုက် သတ်မှတ်ခြင်း
    col_edges = np.linspace(0, img_w, n_cols + 1)

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_items:
            # ဘယ် Column ထဲရောက်သလဲ ရှာဖွေခြင်း
            for c in range(n_cols):
                if col_edges[c] <= item['x'] < col_edges[c+1]:
                    txt = item['text']
                    # Ditto Detection (ပိုမိုကျယ်ပြန့်စွာ စစ်ဆေးခြင်း)
                    is_ditto = any(m in txt for m in ['"', '။', '=', '`', '||', '11', 'LL', 'V', '4', 'U', 'Y', '1', 'I'])
                    
                    if is_ditto and len(txt) <= 2: # စာလုံးတိုမှသာ Ditto အဖြစ်ယူမည်
                        row_cells[c] = "DITTO"
                    else:
                        num = re.sub(r'[^0-9]', '', txt)
                        if num:
                            if c % 2 == 0: # ၃ လုံးဂဏန်းတိုင်
                                row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                            else: # ထိုးကြေးတိုင်
                                row_cells[c] = num
                    break
        final_grid.append(row_cells)

    # --- LOGIC: ဂဏန်း ၃ လုံး နှင့် ထိုးကြေး ခွဲခြားဖြည့်သွင်းခြင်း ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်များတွင်သာ Ditto ဖြည့်မည်
            last_amt = ""
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = last_amt
                elif final_grid[r][c] != "":
                    last_amt = final_grid[r][c]
        else: # ဂဏန်းတိုင်များတွင် Ditto တွေ့ပါက ဖျက်ပစ်မည် (အပေါ်ဂဏန်း မကူးစေရန်)
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = ""
                    
    return final_grid

def save_to_sheets_v7(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").sheet1
        
        # ဒေတာအလွတ်များ ဖယ်ထုတ်ပြီး ပို့ခြင်း
        formatted = [[f"'{c}" if c != "" else "" for c in row] for row in data if any(x != "" for x in row)]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Lottery Scanner v7 (Highest Accuracy)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.write("---")
    st.info("အဆင့်မြှင့်ထားသော အချက်များ -")
    st.write("* ၃ လုံးဂဏန်းများကို အပေါ်အောက် လုံးဝမကူးစေရန် ပိတ်ထားသည်။")
    st.write("* ထိုးကြေးတိုင်တွင်သာ Ditto (။) Logic အလုပ်လုပ်မည်။")
    st.write("* လက်ရေးစောင်းနေမှုကို Row Clustering စနစ်ဖြင့် ပြင်ဆင်ထားသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=450)
    
    if st.button("🔍 ဒေတာအမှန်ထုတ်ယူမည်"):
        with st.spinner("AI စနစ်ဖြင့် စစ်ဆေးနေပါသည်..."):
            res = process_v7(img, a_cols)
            st.session_state['data_v7'] = res

if 'data_v7' in st.session_state:
    st.subheader("စစ်ဆေးရန် ရလဒ်ဇယား")
    # Data Editor တွင် လိုအပ်သည်များကို ကိုယ်တိုင်ပြင်နိုင်ပါသည်
    edited = st.data_editor(st.session_state['data_v7'], use_container_width=True)
    
    if st.button("💾 Google Sheet သို့ အတည်ပြုပို့မည်"):
        if save_to_sheets_v7(edited):
            st.success("ဒေတာအားလုံးကို Sheet ထဲသို့ ထည့်သွင်းပြီးပါပြီ!")
