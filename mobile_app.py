import streamlit as st
import numpy as np
import easyocr
import gspread
import cv2
import re
import json
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- Credentials Setup ---
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
    # အင်္ဂလိပ်စာလုံးနဲ့ ဂဏန်းတွေကို ပိုတိကျအောင် လုပ်ထားပါတယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def expand_r_sorted(text):
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        perms = set([''.join(p) for p in permutations(digits)])
        return sorted(list(perms))
    return [digits.zfill(3)] if digits else []

st.title("🎰 Lottery OCR (Improved Accuracy)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက် (Rows)", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက် (Columns)", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    num_cols = 2 if col_mode == "၂ တိုင်" else (4 if col_mode == "၄ တိုင်" else (6 if col_mode == "၆ တိုင်" else 8))

uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 အကုန်ဖတ်မည် (Auto-fill Mode)"):
        with st.spinner("စာသားများကို တိကျအောင် ဖတ်နေပါသည်..."):
            h, w = img_array.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            col_width = w / num_cols

            for c in range(num_cols):
                crop_img = img_array[0:h, int(c*col_width):int((c+1)*col_width)]
                # allowlist ထည့်ခြင်းဖြင့် ဂဏန်းနဲ့ R ကိုပဲ အဓိကဖတ်စေပါတယ်
                col_results = reader.readtext(crop_img, allowlist='0123456789Rr"\'|/-')
                
                for (bbox, text, prob) in col_results:
                    cy = np.mean([p[1] for p in bbox])
                    r_idx = int((cy / h) * num_rows)
                    if 0 <= r_idx < num_rows:
                        # စာလုံးအမှားများကို ဂဏန်းအဖြစ် ပြောင်းလဲခြင်း
                        cleaned_text = text.upper().replace('I', '1').replace('S', '5').replace('B', '8').replace('G', '6')
                        grid_data[r_idx][c] = cleaned_text.strip()

            # --- Ditto (။) Logic & Cleaning ---
            for c in range(num_cols):
                last_value = ""
                for r in range(num_rows):
                    val = grid_data[r][c]
                    # အမှတ်အသား အမျိုးမျိုးကို စစ်ဆေးခြင်း
                    is_ditto = any(symbol in val for symbol in ["\"", "||", "''", "။", "-", "〃"])
                    
                    if (is_ditto or val == "") and last_value != "":
                        grid_data[r][c] = last_value
                    elif val != "":
                        # ဂဏန်းသန့်သန့်ထုတ်ခြင်း
                        clean_num = re.sub(r'[^0-9Rr]', '', val)
                        if clean_num:
                            grid_data[r][c] = clean_num
                            last_value = clean_num

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန် (အမှားပါက ဒီမှာပြင်ပါ)")
    edited_df = st.data_editor(st.session_state['data_final'], num_rows="dynamic", use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                
                # Sheet 1: Raw Data
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited_df)
                
                # Sheet 2: Expanded Data (R ပါက ဖြန့်ခြင်း)
                sh2 = ss.get_worksheet(1)
                expanded_list = []
                pairs = [(0,1), (2,3), (4,5), (6,7)]
                
                for row in edited_df:
                    for g_col, t_col in pairs:
                        if g_col < len(row) and t_col < len(row):
                            g_val, t_val = str(row[g_col]), str(row[t_col])
                            if g_val and g_val.strip():
                                if 'R' in g_val.upper():
                                    for p in expand_r_sorted(g_val): 
                                        expanded_list.append([p, t_val])
                                else:
                                    clean_num = re.sub(r'\D', '', g_val)
                                    if clean_num: 
                                        expanded_list.append([clean_num[-3:].zfill(3), t_val])
                
                if expanded_list:
                    expanded_list.sort(key=lambda x: x[0])
                    sh2.append_rows(expanded_list)
                
                st.success("🎉 Google Sheet ထဲသို့ အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")