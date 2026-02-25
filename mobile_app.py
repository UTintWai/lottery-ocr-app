import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Precise Amount v6", layout="wide")

@st.cache_resource
def load_ocr():
    # 'en' model ကို သုံးထားပြီး paragraph=False နဲ့ တိကျအောင် လုပ်ပါမယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def pre_process_image(img):
    # စာလုံးတွေ ပိုမည်းလာအောင်နဲ့ ထင်ရှားအောင် လုပ်ခြင်း
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Contrast မြှင့်တင်ခြင်း
    alpha = 1.5 # Contrast control
    beta = 0    # Brightness control
    adjusted = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    return adjusted

def process_v6(img, n_cols):
    h, w = img.shape[:2]
    # ပုံကို အတန်အသင့် ချဲ့ခြင်းက စာလုံးသေးတွေကို ဖတ်ရလွယ်စေပါတယ်
    img_resized = cv2.resize(img, (1500, int(h * (1500 / w))))
    processed_img = pre_process_image(img_resized)
    
    # OCR ဖတ်ခြင်း (detail=1 ပါမှ နေရာအတိအကျရမှာပါ)
    results = reader.readtext(processed_img, detail=1)
    
    data_list = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        data_list.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not data_list: return []

    # --- ROW CLUSTERING ---
    data_list.sort(key=lambda k: k['y'])
    rows_list = []
    current_row = [data_list[0]]
    threshold = 22 # လက်ရေးအတန်းများအတွက် အကွာအဝေးညှိခြင်း

    for i in range(1, len(data_list)):
        if data_list[i]['y'] - current_row[-1]['y'] < threshold:
            current_row.append(data_list[i])
        else:
            rows_list.append(current_row)
            current_row = [data_list[i]]
    rows_list.append(current_row)

    # --- GRID CALCULATION ---
    final_grid = []
    grid_w = processed_img.shape[1]
    col_width = grid_w / n_cols

    for row_data in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        for item in row_data:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                txt = item['text']
                
                # Ditto Recognition (။ သို့မဟုတ် ၎င်းနှင့်တူသော သင်္ကေတများ)
                is_ditto = any(m in txt for m in ['"', '။', '=', '`', '||', '11', 'LL', 'V', '4', 'U', 'Y'])
                
                if is_ditto:
                    row_cells[c_idx] = "DITTO"
                else:
                    # ဂဏန်းနှင့် အက္ခရာများ ရောနေပါက ဂဏန်းကိုသာ ယူခြင်း (ဥပမာ 1800x -> 1800)
                    num = re.sub(r'[^0-9]', '', txt)
                    if num:
                        if c_idx % 2 == 0: # ၃ လုံးဂဏန်းတိုင်
                            row_cells[c_idx] = num.zfill(3) if len(num) <= 3 else num
                        else: # ထိုးကြေးတိုင် (ဂဏန်းအားလုံးကို ယူမည်)
                            row_cells[c_idx] = num
        final_grid.append(row_cells)

    # --- SMART DITTO FILL-DOWN (ထိုးကြေးတိုင်များအတွက်သာ) ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်များ
            last_amount = ""
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = last_amount
                elif final_grid[r][c] != "":
                    last_amount = final_grid[r][c]
        else: # ၃ လုံးဂဏန်းတိုင်များ
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO":
                    final_grid[r][c] = ""
                    
    return final_grid

def save_to_sheets_v6(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open("LotteryData").sheet1
        
        # ' ခံ၍ ပို့ခြင်းဖြင့် 0 များ မပျောက်စေရန် ထိန်းသိမ်းခြင်း
        formatted = [[f"'{c}" if c != "" else "" for c in row] for row in data if any(x != "" for x in row)]
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Precise Lottery Scanner v6")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    st.info("ထိုးကြေးများကို ပိုမိုတိကျစွာဖတ်နိုင်ရန် Contrast နှင့် Detection Logic ကို မြှင့်တင်ထားပါသည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=450, caption="Original Voucher")
    
    if st.button("🔍 Scan & Fill Logic"):
        with st.spinner("ထိုးကြေးများနှင့် Ditto များကို တွက်ချက်နေပါသည်..."):
            res = process_v6(img, a_cols)
            st.session_state['data_v6'] = res

if 'data_v6' in st.session_state:
    edited = st.data_editor(st.session_state['data_v6'], use_container_width=True)
    if st.button("💾 Google Sheet သိမ်းမည်"):
        if save_to_sheets_v6(edited):
            st.success("ဒေတာများ အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
