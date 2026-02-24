import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v3", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM ချွေတာရန် model ကို သီးသန့်သတ်မှတ်
    return easyocr.Reader(['en'], gpu=False, model_storage_directory='models')

reader = load_ocr()

def process_lottery_v3(img, rows, cols):
    # RAM Crash မဖြစ်အောင် ပုံကို အရွယ်အစား အရင်လျှော့မယ်
    h, w = img.shape[:2]
    target_w = 1200
    ratio = target_w / w
    img = cv2.resize(img, (target_w, int(h * ratio)))
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # OCR ဖတ်ခြင်း
    results = reader.readtext(gray)
    
    # Column တွေကို x-coordinate အလိုက် Sort လုပ်ပြီး ခွဲခြားမယ်
    # စာလုံးတွေရဲ့ ဗဟို x မှတ်တွေကို စုစည်းမယ်
    x_coords = sorted([np.mean([p[0] for p in res[0]]) for res in results])
    
    # Clustering logic: x-coordinate တွေကို အုပ်စုဖွဲ့ပြီး တိုင်ခွဲမယ်
    col_boundaries = np.linspace(0, target_w, cols + 1)
    row_boundaries = np.linspace(0, img.shape[0], rows + 1)
    
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        # ဘယ် column/row ထဲကျလဲ ရှာမယ်
        c_idx = np.searchsorted(col_boundaries, cx) - 1
        r_idx = np.searchsorted(row_boundaries, cy) - 1
        
        if 0 <= r_idx < rows and 0 <= c_idx < cols:
            val = text.strip().upper()
            # Ditto Logic
            if any(m in val for m in ['"', '။', '=', 'U', 'V', '`', '4', '||']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', val)
                if num:
                    grid[r_idx][c_idx] = num.zfill(3)

    # Ditto Fill
    for c in range(cols):
        for r in range(1, rows):
            if grid[r][c] == "DITTO" and grid[r-1][c] != "":
                grid[r][c] = grid[r-1][c]
                
    return grid

def save_to_sheets_v3(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        sheet = client.open("LotteryData").sheet1
        
        clean_rows = [r for r in data if any(c != "" for c in r)]
        # Google Sheet မှာ 0 မပျောက်အောင် Quote ခံမယ်
        formatted = [[f"'{c}" if c != "" else "" for c in row] for row in clean_rows]
        
        if formatted:
            sheet.append_rows(formatted)
            return True
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return False

# --- UI ---
st.title("🔢 Lottery Scanner (Precision Mode)")

with st.sidebar:
    a_cols = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    st.write("---")
    st.warning("မှတ်ချက်: ပုံရိုက်လျှင် တည့်တည့်ဖြစ်အောင် ရိုက်ပေးပါ။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)
    
    if st.button("🔍 စတင်ဖတ်ရှုမယ်"):
        with st.spinner("AI မှ ဒေတာများကို ခွဲခြားနေသည်..."):
            res = process_lottery_v3(img, n_rows, a_cols)
            st.session_state['data_v3'] = res

if 'data_v3' in st.session_state:
    edited = st.data_editor(st.session_state['data_v3'], use_container_width=True)
    if st.button("💾 Google Sheet သို့ သိမ်းဆည်းမည်"):
        if save_to_sheets_v3(edited):
            st.success("ဒေတာများ အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
