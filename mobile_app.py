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

st.title("🎰 Lottery OCR (Strict Formatting & Ditto Fix)")

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

    if st.button("🔍 အကုန်ဖတ်မည်"):
        with st.spinner(f"{num_cols} တိုင်စနစ်ဖြင့် တိကျစွာ ဖတ်နေပါသည်..."):
            h, w = img_array.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            results = reader.readtext(img_array)
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx = int((cx / w) * num_cols)
                r_idx = int((cy / h) * num_rows)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    t = text.upper().strip()
                    # Handwriting correction
                    t = t.replace('S', '5').replace('I', '1').replace('Z', '7').replace('B', '8').replace('G', '6').replace('O', '0')
                    grid_data[r_idx][c_idx] = t

            # --- Strict Formatting & Ditto Logic ---
            for c in range(num_cols):
                last_valid_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    # Ditto သင်္ကေတ စစ်ဆေးခြင်း
                    is_ditto = any(s in curr for s in ["\"", "||", "11", "U", "''", "။", "〃", "=", "-", "u"])
                    
                    if (is_ditto or curr == "") and last_valid_val != "":
                        grid_data[r][c] = last_valid_val
                    elif curr != "":
                        # 💡 အတိုင် 1, 3, 5, 7 (ဂဏန်းတိုင်များ)
                        if c % 2 == 0:
                            if 'R' in curr.upper():
                                grid_data[r][c] = curr.upper() # R ပါရင် ဒီအတိုင်းထား
                                last_valid_val = curr.upper()
                            else:
                                clean_g = re.sub(r'\D', '', curr)
                                if clean_g:
                                    formatted_g = clean_g[-3:].zfill(3)
                                    grid_data[r][c] = formatted_g
                                    last_valid_val = formatted_g
                        # 💡 အတိုင် 2, 4, 6, 8 (ထိုးကြေးတိုင်များ)
                        else:
                            clean_t = re.sub(r'\D', '', curr)
                            if clean_t:
                                grid_data[r][c] = clean_t
                                last_valid_val = clean_t

            st.session_state['data_final'] = grid_data
            st.session_state['active_cols'] = num_cols

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_df = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပေါင်းပြီးပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited_df)
                
                # --- Sheet 2: Global Sorting (000-999 အစဉ်လိုက်) ---
                sh2 = ss.get_worksheet(1)
                existing_data = sh2.get_all_values()
                
                master_dict = {}
                # ရှိပြီးသား data ထည့်ခြင်း
                for row in existing_data:
                    if len(row) >= 2 and row[0].isdigit():
                        master_dict[row[0]] = master_dict.get(row[0], 0) + int(row[1])

                # အသစ်ဖတ်လိုက်သော data ပေါင်းထည့်ခြင်း
                active_cols = st.session_state.get('active_cols', 8)
                pairs = [(i, i+1) for i in range(0, active_cols, 2)]
                
                for row in edited_df:
                    for g_col, t_col in pairs:
                        if g_col < len(row) and t_col < len(row):
                            g_val, t_val = str(row[g_col]).strip(), str(row[t_col]).strip()
                            amt = int(t_val) if t_val.isdigit() else 0
                            if g_val:
                                if 'R' in g_val.upper():
                                    for p in expand_r_sorted(g_val):
                                        master_dict[p] = master_dict.get(p, 0) + amt
                                else:
                                    num = re.sub(r'\D', '', g_val)[-3:].zfill(3)
                                    if num.isdigit() and len(num) == 3:
                                        master_dict[num] = master_dict.get(num, 0) + amt

                # ငယ်စဉ်ကြီးလိုက် စီပြီး ပြန်ထည့်ခြင်း
                sorted_list = [[k, master_dict[k]] for k in sorted(master_dict.keys())]
                sh2.clear()
                sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + sorted_list)
                
                st.success("🎉 ဒေတာများကို ပေါင်းစည်းပြီး အစဉ်လိုက် စီပြီးပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")