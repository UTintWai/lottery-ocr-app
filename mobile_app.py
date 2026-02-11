import streamlit as st
import numpy as np
import easyocr
import gspread
import cv2
import re
import json
from PIL import Image
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
from itertools import permutations

# --- Page Setting ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- Google Credentials ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = None
if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
    try:
        secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
        if "private_key" in secret_info:
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
    except Exception as e:
        st.error(f"Credentials Error: {e}")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def expand_r_sorted(text):
    """267R ကို ငယ်စဉ်ကြီးလိုက် ၆ ကွက်ဖြန့်ပေးခြင်း"""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        perms = set([''.join(p) for p in permutations(digits)])
        return sorted(list(perms))
    return [digits.zfill(3)] if digits else []

st.title("🎰 Lottery OCR Pro (Final Updated)")

with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("အတိုင်အရေအတွက် ရွေးပါ", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # 4, 7, 2, 5 တွေ ပိုပီသအောင် contrast မြှင့်ထားသည်
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    st.image(processed, caption="AI ဖတ်မည့်ပုံစံ", use_container_width=True)

    if st.button("🔍 ဒေတာဖတ်မည်"):
        with st.spinner("ဒေတာများကို ခွဲခြားနေပါသည်..."):
            results = reader.readtext(processed)
            h, w = processed.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            y_pts = sorted([res[0][0][1] for res in results])
            top_y, bot_y = (y_pts[0], y_pts[-1]) if y_pts else (0, h)
            cell_h = (bot_y - top_y) / (num_rows if num_rows > 0 else 1)

            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                x_pos = cx / w
                
                # --- Column Logic (Grid System) ---
                if col_mode == "၆ တိုင်": c_idx = min(5, int(x_pos * 6))
                elif col_mode == "၄ တိုင်": c_idx = min(3, int(x_pos * 4))
                elif col_mode == "၂ တိုင်": c_idx = 0 if x_pos < 0.5 else 1
                else: c_idx = min(7, int(x_pos * 8))

                r_idx = int((cy - top_y) // cell_h)
                if 0 <= r_idx < num_rows:
                    # ဂဏန်း၊ R နှင့် ထိုးကြေးကို သေချာဖတ်ခြင်း
                    clean = re.sub(r'[^0-9R]', '', text.upper())
                    grid_data[r_idx][c_idx] = clean
            st.session_state['data'] = grid_data

if 'data' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited = st.data_editor(st.session_state['data'], use_container_width=True, num_rows="dynamic")
    
    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                
                # Sheet 1: Append မူရင်းဒေတာ
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited)
                
                # Sheet 2: ပတ်လည်ဖြန့်ခြင်းနှင့် ငယ်စဉ်ကြီးလိုက်စီခြင်း
                sh2 = ss.get_worksheet(1)
                expanded_list = []
                # အတွဲအတွဲလိုက်ယူခြင်း (ဂဏန်းတိုင်၊ ထိုးကြေးတိုင်)
                if col_mode == "၆ တိုင်": pairs = [(0,1), (2,3), (4,5)]
                elif col_mode == "၄ တိုင်": pairs = [(0,1), (2,3)]
                elif col_mode == "၂ တိုင်": pairs = [(0,1)]
                else: pairs = [(0,1), (2,3), (4,5), (6,7)]

                for row in edited:
                    for g_col, t_col in pairs:
                        g_val = str(row[g_col])
                        t_val = str(row[t_col])
                        if g_val:
                            if 'R' in g_val:
                                for p in expand_r_sorted(g_val): expanded_list.append([p, t_val])
                            else:
                                expanded_list.append([g_val[-3:].zfill(3), t_val])
                
                # Sheet 2 ထဲက ဂဏန်းတွေကို အငယ်ဆုံးကနေ အကြီးဆုံး တစ်ခါတည်း စီသိမ်းမည်
                expanded_list.sort(key=lambda x: x[0])
                
                if expanded_list:
                    sh2.append_rows(expanded_list)
                st.success("🎉 Sheet 1 (Append) နှင့် Sheet 2 (Sorted) သိမ်းပြီးပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")