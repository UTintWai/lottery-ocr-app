import streamlit as st
import numpy as np
import easyocr
import gspread
import cv2
from PIL import Image
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
import re
import json
from itertools import permutations

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- Credentials ---
creds = None
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
    try:
        secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
        if "private_key" in secret_info:
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
    except Exception as e:
        st.error(f"Secret Error: {e}")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def expand_r_sorted(text):
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        perms = set([''.join(p) for p in permutations(digits)])
        return sorted(list(perms))
    return [digits.zfill(3)] if digits else []

st.title("🎰 Lottery OCR (Ditto & Auto-fill)")

# --- Settings Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက် ရွေးပါ", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    
    # 💡 ဒီနေရာမှာ num_cols ကို အသေ သတ်မှတ်ထားလိုက်ပါပြီ (ဒါမှ အောက်က ခလုပ်တွေ အားလုံးမှာ သုံးလို့ရမှာပါ)
    num_cols = 2 if col_mode == "၂ တိုင်" else (4 if col_mode == "၄ တိုင်" else (6 if col_mode == "၆ တိုင်" else 8))

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 အကုန်ဖတ်မည် (Auto-fill Mode)"):
        with st.spinner("။ များကို အလိုအလျောက် ဖြည့်စွက်နေပါသည်..."):
            h, w = img_array.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            col_width = w / num_cols

            for c in range(num_cols):
                crop_img = img_array[0:h, int(c*col_width):int((c+1)*col_width)]
                col_results = reader.readtext(crop_img)
                
                for (bbox, text, prob) in col_results:
                    cy = np.mean([p[1] for p in bbox])
                    r_idx = int((cy / h) * num_rows)
                    if 0 <= r_idx < num_rows:
                        grid_data[r_idx][c] = text.strip()

            for c in range(num_cols):
                last_value = ""
                for r in range(num_rows):
                    current_val = grid_data[r][c]
                    if current_val in ["။", "။။", "〃", "''", ""] and last_value != "":
                        grid_data[r][c] = last_value
                    
                    clean_val = re.sub(r'[^0-9Rr]', '', grid_data[r][c].upper())
                    if clean_val:
                        grid_data[r][c] = clean_val
                        last_value = clean_val

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_df = st.data_editor(st.session_state['data_final'], num_rows="dynamic", use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData") # Sheet နာမည် မှန်အောင် စစ်ပေးပါ
                
                # ပထမ Sheet ထဲ ထည့်ခြင်း
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited_df)
                
                # ဒုတိယ Sheet (Expanded Data) ထဲ ထည့်ခြင်း
                sh2 = ss.get_worksheet(1)
                expanded_list = []
                
                # num_cols ကို အပေါ်ကနေ ယူသုံးထားပါတယ်
                pairs = [(0,1), (2,3), (4,5), (6,7)] if num_cols == 8 else ([(0,1), (2,3), (4,5)] if num_cols == 6 else ([(0,1), (2,3)] if num_cols == 4 else [(0,1)]))

                for row in edited_df:
                    for g_col, t_col in pairs:
                        # Index ကျော်မသွားအောင် စစ်ဆေးခြင်း
                        if g_col < len(row) and t_col < len(row):
                            g_val, t_val = str(row[g_col]), str(row[t_col])
                            if g_val:
                                if 'R' in g_val.upper():
                                    for p in expand_r_sorted(g_val): expanded_list.append([p, t_val])
                                else:
                                    clean_num = re.sub(r'\D', '', g_val)
                                    if clean_num: expanded_list.append([clean_num[-3:].zfill(3), t_val])
                
                expanded_list.sort(key=lambda x: x[0])
                if expanded_list: 
                    sh2.append_rows(expanded_list)
                
                st.success("🎉 သိမ်းဆည်းမှု အောင်မြင်ပါသည်။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")