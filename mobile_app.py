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

def expand_r(text):
    """267R ကို ၆ ကွက်ဖြန့်ပေးခြင်း"""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        return sorted(list(set([''.join(p) for p in permutations(digits)])))
    return [digits]

st.title("🎰 Lottery OCR (Sheet 1 & 2 System)")

with st.sidebar:
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    img_np = np.array(img)
    
    # 7, 4 ပိုပီသအောင် အဖြူအမဲ ပြတ်အောင်လုပ်ခြင်း
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    st.image(processed, caption="AI ဖတ်မည့်ပုံစံ", use_container_width=True)

    if st.button("🔍 ဒေတာဖတ်မည်"):
        results = reader.readtext(processed)
        h, w = processed.shape[:2]
        grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
        y_pts = sorted([res[0][0][1] for res in results])
        top_y, bot_y = (y_pts[0], y_pts[-1]) if y_pts else (0, h)
        cell_h = (bot_y - top_y) / num_rows

        for (bbox, text, prob) in results:
            cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
            x_pos = cx / w
            # Column logic (Simplified)
            c_idx = int(x_pos * 8) if col_mode == "၈ တိုင်" else (0 if x_pos < 0.5 else 1)
            r_idx = int((cy - top_y) // cell_h)
            if 0 <= r_idx < num_rows:
                # ဂဏန်းနဲ့ R ကိုပဲ သိမ်းမယ်
                clean = re.sub(r'[^0-9Rr]', '', text.upper())
                grid_data[r_idx][c_idx] = clean
        st.session_state['data'] = grid_data

if 'data' in st.session_state:
    edited = st.data_editor(st.session_state['data'])
    if st.button("💾 Sheet သို့ အားလုံးသိမ်းမည်"):
        if creds:
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1: အဟောင်းအောက်မှာ အသစ်ဆက်သိမ်း (Append)
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited)
            
            # Sheet 2: ပတ်လည်ကွက်များ ဖြန့်သိမ်းခြင်း
            sh2 = ss.get_worksheet(1)
            r_list = []
            for row in edited:
                for val in row:
                    if 'R' in str(val):
                        for p in expand_r(str(val)): r_list.append([p])
                    elif str(val).isdigit() and len(str(val)) == 3:
                        r_list.append([val])
            if r_list: sh2.append_rows(r_list)
            st.success("🎉 Sheet 1 (Append) နှင့် Sheet 2 (R-Expanded) သိမ်းပြီးပါပြီ!")