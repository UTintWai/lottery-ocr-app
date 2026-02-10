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
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        perms = set([''.join(p) for p in permutations(digits)])
        return sorted(list(perms))
    return [digits.zfill(3)] if digits else []

st.title("🎰 Lottery OCR Pro (Final System)")

# --- Sidebar (အတိုင်ရွေးစရာ ပြန်ထည့်ပေးထားပါသည်) ---
with st.sidebar:
    st.header("⚙️ Settings")
    col_mode = st.selectbox("အတိုင်အရေအတွက် ရွေးပါ", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    st.image(processed, caption="AI ဖတ်မည့်ပုံစံ", use_container_width=True)

    if st.button("🔍 ဒေတာဖတ်မည်"):
        results = reader.readtext(processed)
        results.sort(key=lambda x: x[0][0][1]) # Y-sort
        
        extracted_data = []
        for i, (bbox, text, prob) in enumerate(results):
            clean_text = re.sub(r'[^0-9R]', '', text.upper())
            if len(re.sub(r'\D', '', clean_text)) == 3:
                num_val = clean_text
                bet_val = "0"
                curr_y = np.mean([p[1] for p in bbox])
                curr_x_end = max([p[0] for p in bbox])
                
                for next_bbox, next_text, next_prob in results:
                    next_y = np.mean([p[1] for p in next_bbox])
                    next_x_start = min([p[0] for p in next_bbox])
                    if abs(curr_y - next_y) < 25 and 0 < (next_x_start - curr_x_end) < 200:
                        bet_val = re.sub(r'\D', '', next_text)
                        break
                extracted_data.append([num_val, bet_val])

        # ဂဏန်းတိုင်အလိုက် အငယ်မှအကြီး စီပေးခြင်း
        extracted_data.sort(key=lambda x: x[0])
        st.session_state['data'] = extracted_data

if 'data' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    # အတိုင်ရွေးချယ်မှုအပေါ်မူတည်ပြီး Table ကို ပြသခြင်း
    edited = st.data_editor(st.session_state['data'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Google Sheet သို့ သိမ်းမည်"):
        if creds:
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # Sheet 1
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited)
            
            # Sheet 2 (ပတ်လည်ဖြန့်ပြီး Sorting စီခြင်း)
            sh2 = ss.get_worksheet(1)
            final_expanded = []
            for num, bet in edited:
                if 'R' in num:
                    for p in expand_r_sorted(num): final_expanded.append([p, bet])
                else:
                    final_expanded.append([num[:3].zfill(3), bet])
            
            final_expanded.sort(key=lambda x: x[0]) # အငယ်ဆုံးမှ အကြီးဆုံး စီသည်
            if final_expanded:
                sh2.append_rows(final_expanded)
            st.success("🎉 သိမ်းဆည်းမှု အောင်မြင်ပါသည်။")