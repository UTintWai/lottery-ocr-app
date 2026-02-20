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

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Lottery Pro 2026 Precise", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_digit_templates():
    templates = {}
    temp_path = "templates"
    if os.path.exists(temp_path):
        for filename in os.listdir(temp_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(temp_path, filename), 0)
                if img is not None:
                    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                    digit_name = filename.split('.')[0].split('_')[0]
                    templates[digit_name] = cv2.resize(img, (28, 28))
    return templates

reader = load_ocr()
digit_templates = load_digit_templates()

# ---------------- 2. IMPROVED SEGMENTATION ----------------
def segment_and_match(roi_gray):
    if not digit_templates: return None
    
    # Noise လျှော့ချပြီး စာလုံးကို ပေါ်အောင်လုပ်ခြင်း
    processed = cv2.medianBlur(roi_gray, 3)
    _, thresh = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # စာလုံးတွေ ပူးနေရင် ခွဲဖို့ Erosion အနည်းငယ် သုံးခြင်း
    kernel = np.ones((2,2), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digit_list = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        # ဂဏန်း ဖြစ်နိုင်ချေရှိတဲ့ အရွယ်အစားကို စစ်ထုတ်ခြင်း (အရမ်းသေးရင် ဖယ်မယ်)
        if h > 12 and w > 4: 
            digit_list.append((x, y, w, h))
    
    # ဘယ်မှညာ အစဉ်လိုက် စီမယ်
    digit_list.sort(key=lambda x: x[0])
    
    final_string = ""
    for x, y, w, h in digit_list:
        digit_roi = thresh[y:y+h, x:x+w]
        # ဘေးပတ်လည် space အနည်းငယ် ထည့်ခြင်း
        digit_roi = cv2.copyMakeBorder(digit_roi, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        digit_roi = cv2.resize(digit_roi, (28, 28))
        
        best_score = -1
        best_digit = ""
        for name, temp_img in digit_templates.items():
            res = cv2.matchTemplate(digit_roi, temp_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_digit = name
        
        if best_score > 0.35: # အနည်းဆုံး ၃၅ ရာခိုင်နှုန်း တူမှ ယူမယ်
            final_string += best_digit
                
    return final_string if final_string else None

def process_bet_logic(num_txt, amt_txt):
    num_clean = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_clean = re.sub(r'[^0-9]', '', str(amt_txt))
    amt = int(amt_clean) if amt_clean else 0
    results = {}
    if 'R' in num_clean:
        base = num_clean.replace('R', '')
        perms = sorted(list(set([''.join(p) for p in permutations(base)]))) if len(base) == 3 else [base]
        if perms and amt > 0:
            split = amt // len(perms)
            for p in perms: results[p] = split
    elif num_clean:
        results[num_clean.zfill(3) if num_clean.isdigit() else num_clean] = amt
    return results

# ---------------- 3. UI & SCAN LOOP ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    active_cols = st.selectbox("Select Columns", [2, 4, 6, 8], index=2)
    num_rows = st.number_input("Select Rows", min_value=1, value=25)

uploaded_file = st.file_uploader("Upload Voucher", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 Scan Voucher"):
        with st.spinner("ဂဏန်းများကို တိကျစွာ ဖတ်နေသည်..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            col_w = w / active_cols
            grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]

            # EasyOCR နဲ့ စာသားတည်နေရာ အရင်ရှာမယ်
            results = reader.readtext(gray)
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx, r_idx = int(cx // col_w), int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < active_cols:
                    x_min, y_min = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
                    x_max, y_max = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
                    roi = gray[max(0,y_min-2):min(h,y_max+2), max(0,x_min-2):min(w,x_max+2)]
                    
                    # ဂဏန်းတိုင် (Number Column) အတွက် Template Matching ကို အဓိကသုံးမယ်
                    if c_idx % 2 == 0:
                        custom_text = segment_and_match(roi)
                        final_text = custom_text if custom_text else text
                    else:
                        final_text = text

                    clean_txt = re.sub(r'[^0-9R]', '', str(final_text).upper())
                    grid_data[r_idx][c_idx] = clean_txt

            st.session_state['data_final'] = grid_data
            st.success("✅ Scanning Complete!")

# ---------------- 4. DISPLAY & SHEET ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Send to Google Sheet"):
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh1, sh2 = ss.get_worksheet(0), ss.get_worksheet(1)
            sh1.append_rows(edited_data)

            # Summary Process
            master_sum = {}
            for row in edited_data:
                for i in range(0, len(row)-1, 2):
                    if row[i] and row[i+1]:
                        res = process_bet_logic(row[i], row[i+1])
                        for k, v in res.items(): master_sum[k] = master_sum.get(k, 0) + v

            sh2.clear()
            final_summary = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["Number", "Total"]] + final_summary)
            st.success("✅ Data sent successfully!")
        except Exception as e:
            st.error(f"Error: {e}")
