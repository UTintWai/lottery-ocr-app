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

# --- Google Credentials ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = None
if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
    secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
    creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)

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

st.title("🎰 Lottery OCR Pro (Final Alignment)")

with st.sidebar:
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 7, 4 ပိုပီသစေရန် Contrast မြှင့်တင်ခြင်း
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    st.image(processed, caption="AI ဖတ်မည့်ပုံစံ", use_container_width=True)

    if st.button("🔍 ဒေတာဖတ်မည်"):
        results = reader.readtext(processed)
        h, w = processed.shape[:2]
        grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
        
        y_pts = sorted([res[0][0][1] for res in results])
        top_y, bot_y = (y_pts[0], y_pts[-1]) if y_pts else (0, h)
        cell_h = (bot_y - top_y) / (num_rows if num_rows > 0 else 1)

        for (bbox, text, prob) in results:
            cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
            x_pos = cx / w
            
            # --- ၆ တိုင်အတွက် တိကျသော Boundary (အစ်ကို့စာမူအချိုးအစား) ---
            if col_mode == "၆ တိုင်":
                if x_pos < 0.16: c_idx = 0      # ဂဏန်း ၁
                elif x_pos < 0.33: c_idx = 1    # ထိုးကြေး ၁
                elif x_pos < 0.50: c_idx = 2    # ဂဏန်း ၂
                elif x_pos < 0.66: c_idx = 3    # ထိုးကြေး ၂
                elif x_pos < 0.83: c_idx = 4    # ဂဏန်း ၃
                else: c_idx = 5                 # ထိုးကြေး ၃
            elif col_mode == "၂ တိုင်":
                c_idx = 0 if x_pos < 0.45 else 1
            elif col_mode == "၄ တိုင်":
                if x_pos < 0.22: c_idx = 0
                elif x_pos < 0.45: c_idx = 1
                elif x_pos < 0.72: c_idx = 2
                else: c_idx = 3
            else: # ၈ တိုင်
                c_idx = min(7, int(x_pos * 8))

            r_idx = int((cy - top_y) // cell_h)
            if 0 <= r_idx < num_rows:
                clean = re.sub(r'[^0-9Rr]', '', text.upper())
                grid_data[r_idx][c_idx] = clean
        st.session_state['data'] = grid_data

if 'data' in st.session_state:
    edited = st.data_editor(st.session_state['data'], use_container_width=True)
    if st.button("💾 Google Sheet သို့ အားလုံးသိမ်းမည်"):
        if creds:
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: Append (မူရင်းအတိုင်း)
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited)
            
            # Sheet 2: Expanded & Sorted (ဂဏန်း + ထိုးကြေး)
            sh2 = ss.get_worksheet(1)
            expanded_list = []
            
            # အတိုင်အတွဲများ (ဂဏန်းတိုင်, ထိုးကြေးတိုင်)
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
                            for p in expand_r_sorted(g_val):
                                expanded_list.append([p, t_val])
                        else:
                            # ၃ လုံးပြည့်အောင်ညှိပြီး ဂဏန်းနှင့်ထိုးကြေးအတွဲလိုက်သိမ်းမည်
                            expanded_list.append([g_val[-3:].zfill(3), t_val])
            
            if expanded_list:
                sh2.append_rows(expanded_list)
            st.success("🎉 သိမ်းဆည်းမှုအောင်မြင်ပါသည်။")