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
    # 'en' (အင်္ဂလိပ်စာ) ပါထည့်ထားမှ R ကို ဖတ်နိုင်မှာပါ
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def expand_r_sorted(text):
    """267R ကို ငယ်စဉ်ကြီးလိုက် ၆ ကွက်ဖြန့်ပေးခြင်း"""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        # Permutations လုပ်ပြီး set နဲ့ duplicate ဖယ်၊ ပြီးမှ sorting စီသည်
        perms = set([''.join(p) for p in permutations(digits)])
        return sorted(list(perms))
    return [digits.zfill(3)] if digits else []

st.title("🎰 Lottery OCR Pro (Final Fix)")

uploaded_file = st.file_uploader("ပုံတင်ရန် (JPG, PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # 7, 4 နဲ့ R ပိုပီသအောင် Image Contrast မြှင့်တင်ခြင်း
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    st.image(processed, caption="AI ဖတ်မည့်ပုံစံ (Contrast မြှင့်ထားသည်)", use_container_width=True)

    if st.button("🔍 ဒေတာအားလုံးကို ဖတ်မည်"):
        # paragraph=False ထားမှ တစ်လုံးချင်းစီရဲ့ တည်နေရာကို တိတိကျကျ ရမှာပါ
        results = reader.readtext(processed, detail=1)
        
        # စာသားများကို အပေါ်မှအောက် (Y အလိုက်) အရင်စီမည်
        results.sort(key=lambda x: x[0][0][1])
        
        extracted_data = []
        for i, (bbox, text, prob) in enumerate(results):
            # စာသားထဲက ဂဏန်းနဲ့ R ကိုပဲ ယူမည်
            clean_text = re.sub(r'[^0-9R]', '', text.upper())
            
            # ဂဏန်း ၃ လုံး (သို့) R ပါတဲ့ ဂဏန်းတွေ့ရင် (ဥပမာ 123 သို့မဟုတ် 123R)
            if len(re.sub(r'\D', '', clean_text)) == 3:
                num_val = clean_text
                bet_val = "0" # Default ထိုးကြေး
                
                # သူ့ရဲ့ ညာဘက်အနီးဆုံးမှာ ရှိတဲ့ စာသားကို ထိုးကြေးအဖြစ် ယူမည်
                curr_y = np.mean([p[1] for p in bbox])
                curr_x_end = max([p[0] for p in bbox])
                
                for next_bbox, next_text, next_prob in results:
                    next_y = np.mean([p[1] for p in next_bbox])
                    next_x_start = min([p[0] for p in next_bbox])
                    
                    # စာကြောင်းတစ်ကြောင်းတည်းဖြစ်ပြီး ညာဘက် 150 pixel အတွင်းရှိနေရင်
                    if abs(curr_y - next_y) < 25 and 0 < (next_x_start - curr_x_end) < 150:
                        bet_val = re.sub(r'\D', '', next_text)
                        break
                
                extracted_data.append([num_val, bet_val])

        # ဒေတာများကို ဂဏန်းတိုင်အလိုက် ငယ်စဉ်ကြီးလိုက် စီပေးခြင်း (Sorting)
        extracted_data.sort(key=lambda x: x[0])
        st.session_state['data'] = extracted_data

if 'data' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited = st.data_editor(st.session_state['data'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Google Sheet သို့ သိမ်းမည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                
                # Sheet 1: မူရင်းဒေတာများကို သိမ်းမည်
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited)
                
                # Sheet 2: ပတ်လည်ဖြန့်ပြီး ငယ်စဉ်ကြီးလိုက် သိမ်းမည်
                sh2 = ss.get_worksheet(1)
                final_expanded = []
                for num, bet in edited:
                    if 'R' in num:
                        for p in expand_r_sorted(num):
                            final_expanded.append([p, bet])
                    else:
                        final_expanded.append([num[:3].zfill(3), bet])
                
                # Sheet 2 ထဲက ဒေတာများကိုပါ ငယ်စဉ်ကြီးလိုက် တစ်ခါပြန်စီမည်
                final_expanded.sort(key=lambda x: x[0])
                
                if final_expanded:
                    sh2.append_rows(final_expanded)
                st.success("🎉 သိမ်းဆည်းမှု အောင်မြင်ပါသည်။ (Sheet 2 တွင် ငယ်စဉ်ကြီးလိုက် စီပြီးပါပြီ)")
            except Exception as e:
                st.error(f"Sheet Error: {e}")