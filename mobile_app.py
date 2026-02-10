import streamlit as st
import numpy as np
import easyocr
import gspread
import cv2
import re
from PIL import Image
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
import json

# --- Page Setting ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- Google Credentials ---
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
creds = None
if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
    secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
    if "private_key" in secret_info:
        secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)

@st.cache_resource
def load_ocr():
    # 'en' စာလုံးပါ ဖတ်ခိုင်းထားသည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def get_permutations(text):
    """267R ကို ၆ ကွက် ဖြန့်ပေးသော function"""
    nums = re.sub(r'\D', '', text)
    if len(nums) != 3: return [nums]
    from itertools import permutations
    return sorted(list(set([''.join(p) for p in permutations(nums)])))

st.title("🎰 Lottery OCR Pro (R-System)")

with st.sidebar:
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # --- Image Processing (4, 7 ပိုပီသစေရန်) ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    st.image(processed_img, caption="AI ဖတ်မည့်ပုံစံ", use_container_width=True)

    if st.button("🔍 ဒေတာဖတ်မည်"):
        results = reader.readtext(processed_img)
        h, w = processed_img.shape[:2]
        grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
        
        y_pts = sorted([res[0][0][1] for res in results])
        top_y, bot_y = (y_pts[0], y_pts[-1]) if y_pts else (0, h)
        cell_h = (bot_y - top_y) / num_rows

        for (bbox, text, prob) in results:
            cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
            x_pos = cx / w
            
            # --- Column Logic ---
            if col_mode == "၂ တိုင်": c_idx = 0 if x_pos < 0.45 else 1
            elif col_mode == "၄ တိုင်":
                if x_pos < 0.22: c_idx = 0
                elif x_pos < 0.45: c_idx = 1
                elif x_pos < 0.72: c_idx = 2
                else: c_idx = 3
            else: c_idx = min(7, int(x_pos * 8))

            r_idx = int((cy - top_y) // cell_h)
            if 0 <= r_idx < num_rows:
                # ဂဏန်း နှင့် R ကိုပဲ ယူမည်
                clean = re.sub(r'[^0-9Rr]', '', text.upper())
                grid_data[r_idx][c_idx] = clean

        st.session_state['current_page'] = grid_data

if 'current_page' in st.session_state:
    edited_data = st.data_editor(st.session_state['current_page'], use_container_width=True)
    
    if st.button("💾 Google Sheet သို့ သိမ်းမည်"):
        if creds:
            client = gspread.authorize(creds)
            spreadsheet = client.open("LotteryData")
            
            # Sheet 1: မူရင်းဒေတာများကို Append လုပ်ခြင်း
            sh1 = spreadsheet.get_worksheet(0)
            sh1.append_rows(edited_data)
            
            # Sheet 2: ပတ်လည်ကွက်များကို ဖြန့်သိမ်းခြင်း
            sh2 = spreadsheet.get_worksheet(1)
            flat_list = []
            for r in edited_data:
                for c_idx in [0, 2, 4, 6]: # ဂဏန်းတိုင်များ
                    val = str(r[c_idx])
                    if 'R' in val:
                        perms = get_permutations(val)
                        for p in perms: flat_list.append([p])
                    elif val:
                        flat_list.append([val[-3:].zfill(3)])
            
            if flat_list:
                sh2.append_rows(flat_list)
                
            st.success("🎉 Sheet 1 တွင် ဒေတာအသစ်ပေါင်းထည့်ပြီး၊ Sheet 2 တွင် ပတ်လည်ကွက်များ ဖြန့်ပြီးပါပြီ!")