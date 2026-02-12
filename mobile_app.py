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

st.title("🎰 Lottery OCR (Fixed Columns & Accuracy)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက် (Rows)", min_value=1, value=25)
    # ၈ တိုင်ကို ပုံသေထားပြီး ဖတ်ခိုင်းပါမယ်
    st.info("အကောင်းဆုံးရလဒ်အတွက် ၈ တိုင် (8 Columns) mode ကို သုံးထားပေးပါတယ်")
    num_cols = 8

uploaded_file = st.file_uploader("လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_array = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, use_container_width=True)

    if st.button("🔍 အကုန်ဖတ်မည် (Auto-fill Mode)"):
        with st.spinner("စာသားများကို တိကျအောင် တွက်ချက်နေပါသည်..."):
            h, w = img_array.shape[:2]
            # ၈ တိုင်အတွက် grid ဆောက်ခြင်း
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            col_width = w / 8

            for c in range(8):
                crop_img = img_array[0:h, int(c*col_width):int((c+1)*col_width)]
                # OCR accuracy တက်စေရန် Contrast မြှင့်ခြင်း
                gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                results = reader.readtext(gray)
                
                for (bbox, text, prob) in results:
                    # စာသား၏ အမြင့်တည်နေရာကို တွက်ချက်ခြင်း
                    cy = np.mean([p[1] for p in bbox])
                    r_idx = int((cy / h) * num_rows)
                    if 0 <= r_idx < num_rows:
                        # စာလုံးအမှားများကို ဂဏန်းသို့ အလိုအလျောက် ပြောင်းလဲခြင်း
                        t = text.upper().strip()
                        t = t.replace('S', '5').replace('I', '1').replace('Z', '7').replace('B', '8').replace('G', '6').replace('O', '0')
                        grid_data[r_idx][c] = t

            # --- ။ (Ditto) & Auto-fill Logic ---
            for c in range(8):
                last_val = ""
                for r in range(num_rows):
                    curr = grid_data[r][c]
                    # Ditto ဖြစ်နိုင်သော သင်္ကေတများကို စစ်ဆေးခြင်း
                    is_ditto = any(s in curr for s in ["\"", "||", "11", "U", "''", "။", "〃", "=", "-"])
                    
                    if (is_ditto or curr == "") and last_val != "":
                        grid_data[r][c] = last_val
                    elif curr != "":
                        # ဂဏန်းနှင့် R သာယူရန်
                        clean = re.sub(r'[^0-9Rr]', '', curr)
                        if clean:
                            grid_data[r][c] = clean
                            last_val = clean

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    # အောက်ပါ editor တွင် လွဲနေသော အကွက်များကို တိုက်ရိုက် ပြင်နိုင်ပါသည်
    edited_df = st.data_editor(st.session_state['data_final'], num_rows="dynamic", use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                
                # Sheet 1: အကွက်လိုက် သိမ်းခြင်း
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited_df)
                
                # Sheet 2: Expanded Data (R ပါက ဖြန့်ထုတ်ခြင်း)
                sh2 = ss.get_worksheet(1)
                expanded_list = []
                # ၈ တိုင်အတွက် (ဂဏန်း၊ ထိုးကြေး) တွဲဖက်မှုများ
                pairs = [(0,1), (2,3), (4,5), (6,7)]
                
                for row in edited_df:
                    for g_col, t_col in pairs:
                        if g_col < len(row) and t_col < len(row):
                            g_val, t_val = str(row[g_col]), str(row[t_col])
                            if g_val.strip():
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
                
                st.success("🎉 အချက်အလက်များ Google Sheet သို့ အောင်မြင်စွာ ပို့ပြီးပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")