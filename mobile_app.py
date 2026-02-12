import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

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

st.title("🎰 Lottery OCR (Full Logic & Sheet 2 Fix)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက် (Rows)", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက် ရွေးပါ", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    num_cols = int(col_mode.split()[0])

uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 အကုန်ဖတ်မည် (Fast Mode)"):
        with st.spinner(f"{num_cols} တိုင်စနစ်ဖြင့် ဖတ်နေပါသည်..."):
            h, w = img_array.shape[:2]
            # 💡 ၈ တိုင်စလုံးအတွက် grid ကြိုဆောက်ထားပါမယ် (Spreadsheet သဏ္ဌာန်တူစေရန်)
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            results = reader.readtext(img_array)
            
            for (bbox, text, prob) in results:
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                # 💡 ရွေးထားတဲ့ Column အရေအတွက်ပေါ်မူတည်ပြီး အကွက်ရှာခြင်း
                c_idx = int((cx / w) * num_cols)
                r_idx = int((cy / h) * num_rows)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    t = text.upper().strip()
                    # Handwriting Correction
                    t = t.replace('S', '5').replace('I', '1').replace('Z', '7').replace('B', '8').replace('G', '6').replace('O', '0')
                    grid_data[r_idx][c_idx] = t

            # --- Ditto (။) Logic ---
            for c in range(num_cols):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c])
                    is_ditto = any(s in curr for s in ["\"", "||", "11", "U", "''", "။", "〃", "=", "-"])
                    if (is_ditto or curr == "") and last_val != "":
                        grid_data[r][c] = last_val
                    elif curr != "":
                        clean = re.sub(r'[^0-9Rr]', '', curr)
                        if clean:
                            grid_data[r][c] = clean
                            last_val = clean

            st.session_state['data_final'] = grid_data
            st.session_state['current_cols'] = num_cols

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_df = st.data_editor(st.session_state['data_final'], num_rows="dynamic", use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပေါင်းပြီးပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                
                # --- Sheet 1: Raw Data သိမ်းခြင်း ---
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited_df)
                
                # --- Sheet 2: ထိုးကြေးပေါင်းခြင်း Logic (အပြည့်အစုံ) ---
                summary_dict = {}
                current_cols = st.session_state.get('current_cols', 8)
                # (ဂဏန်း၊ ထိုးကြေး) အတွဲများကို ရှာဖွေခြင်း
                pairs = [(i, i+1) for i in range(0, current_cols, 2)]
                
                for row in edited_df:
                    for g_col, t_col in pairs:
                        if g_col < len(row) and t_col < len(row):
                            g_val = str(row[g_col]).strip()
                            t_val_raw = str(row[t_col]).strip()
                            
                            # ထိုးကြေးကို ဂဏန်းသက်သက်ပြောင်း (ဥပမာ "24." -> 24)
                            t_val_clean = re.sub(r'\D', '', t_val_raw)
                            t_amount = int(t_val_clean) if t_val_clean else 0
                            
                            if g_val:
                                # R ပါလျှင် ဖြန့်ထုတ်ပေါင်းမည်
                                if 'R' in g_val.upper():
                                    for p in expand_r_sorted(g_val):
                                        summary_dict[p] = summary_dict.get(p, 0) + t_amount
                                else:
                                    # ဂဏန်းသက်သက်ပဲယူပြီး ၃ လုံးစနစ်ပြောင်းမည်
                                    clean_g = re.sub(r'\D', '', g_val)
                                    if clean_g:
                                        num_key = clean_g[-3:].zfill(3)
                                        summary_dict[num_key] = summary_dict.get(num_key, 0) + t_amount
                
                # --- Sheet 2 ထဲသို့ ဒေတာသွင်းခြင်း ---
                sh2 = ss.get_worksheet(1)
                # Dictionary ကို List ပြောင်းပြီး ဂဏန်းအစဉ်လိုက် စီမည်
                final_list = [[k, v] for k, v in summary_dict.items() if v > 0]
                final_list.sort(key=lambda x: x[0])
                
                if final_list:
                    sh2.append_rows(final_list)
                
                st.success(f"🎉 အောင်မြင်ပါသည်။ {len(final_list)} မျိုးကို Sheet 2 ထဲသို့ ပေါင်းထည့်လိုက်ပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")