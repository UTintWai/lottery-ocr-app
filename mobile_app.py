import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import os
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026 Stable OCR", layout="wide")

# ---------------- 1. OCR & TEMPLATES LOAD ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_digit_templates():
    templates = {}
    # သင့် VS Code ထဲက 'templates' folder လမ်းကြောင်း
    temp_path = "templates" 
    if os.path.exists(temp_path):
        for i in range(10):
            # ပုံစံမျိုးစုံ ရှိနိုင်လို့ .png ရော .jpg ရော စစ်ပါတယ်
            p = os.path.join(temp_path, f"{i}.png")
            if not os.path.exists(p): p = os.path.join(temp_path, f"{i}.jpg")
            
            if os.path.exists(p):
                img = cv2.imread(p, 0)
                # ပုံကို အဖြူအမည်း ပြောင်းပြန်လှန်ပြီး binary လုပ်ပါတယ်
                _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                templates[str(i)] = cv2.resize(img, (28, 28))
    return templates

reader = load_ocr()
digit_templates = load_digit_templates()

# ---------------- 2. TEMPLATE MATCHING LOGIC ----------------
def match_digit_from_image(roi_gray):
    """ ပုံရိပ်လေးကို ဂဏန်းအဖြစ် ပြောင်းပေးတဲ့ function """
    if not digit_templates: return None
    
    # ပုံကို သန့်စင်အောင်လုပ်ခြင်း
    roi = cv2.resize(roi_gray, (28, 28))
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    
    best_score = -1
    best_digit = None
    
    for digit, temp_img in digit_templates.items():
        res = cv2.matchTemplate(roi, temp_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_digit = digit
            
    return best_digit if best_score > 0.5 else None

# ---------------- 3. BET LOGIC ----------------
def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X', '*')
    results = {}
    try:
        if 'R' in clean_num:
            base = clean_num.replace('R', '')
            perms = sorted(list(set([''.join(p) for p in permutations(base)]))) if len(base)==3 else [base]
            match_amt = re.findall(r'\d+', amt_str)
            amt = int(match_amt[0]) if match_amt else 0
            if perms and amt > 0:
                split = amt // len(perms)
                for p in perms: results[p] = split
        else:
            match_amt = re.findall(r'\d+', amt_str)
            amt = int(match_amt[0]) if match_amt else 0
            num_final = clean_num.zfill(3) if clean_num.isdigit() else clean_num
            if num_final: results[num_final] = amt
    except: pass
    return results

# ---------------- 4. MAIN UI ----------------
uploaded_file = st.file_uploader("Upload Voucher", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 Scan & Match Digits"):
        with st.spinner("ဖတ်နေသည်..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Settings
            num_rows = 25
            num_cols = 8
            col_w = w / num_cols
            grid = [["" for _ in range(num_cols)] for _ in range(num_rows)]

            # EasyOCR နဲ့ အရင်ဖတ်မယ်
            results = reader.readtext(gray)

            for (bbox, text, prob) in results:
                # နေရာတွက်ချက်ခြင်း
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                c_idx, r_idx = int(cx / col_w), int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    # OCR စိတ်မချရရင် (သို့) ဂဏန်းပုံစံဖြစ်နေရင် Template နဲ့ ထပ်စစ်မယ်
                    if prob < 0.6: 
                        x_min, y_min = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
                        x_max, y_max = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
                        roi = gray[y_min:y_max, x_min:x_max]
                        matched = match_digit_from_image(roi)
                        if matched: text = matched

                    # Cleanup
                    txt = re.sub(r'[^0-9R]', '', text.upper())
                    if c_idx % 2 == 0: # ဂဏန်းတိုင်
                        grid[r_idx][c_idx] = txt.zfill(3) if txt.isdigit() else txt
                    else: # ငွေပမာဏတိုင်
                        nums = re.findall(r'\d+', text)
                        grid[r_idx][c_idx] = nums[0] if nums else ""

            st.session_state['data'] = grid
            st.success("✅ ဖတ်လို့ပြီးပါပြီ။ လိုအပ်တာ ပြင်နိုင်ပါတယ်။")

# ---------------- 5. GOOGLE SHEET UPLOAD ----------------
if 'data' in st.session_state:
    edited = st.data_editor(st.session_state['data'], use_container_width=True)
    
    if st.button("🚀 Upload to Sheets"):
        try:
            # GCP Secrets အသုံးပြုခြင်း
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)

            ss = client.open("LotteryData") # သင့် Sheet အမည်
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited.tolist() if isinstance(edited, np.ndarray) else edited)
            
            st.success("✅ Google Sheet ထဲသို့ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
