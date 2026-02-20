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
st.set_page_config(page_title="Lottery Pro 2026 Stable", layout="wide")

# ---------------- 1. LOAD OCR & TEMPLATES ----------------
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
                    digit_name = filename.split('.')[0]
                    templates[digit_name] = cv2.resize(img, (28, 28))
    return templates

reader = load_ocr()
digit_templates = load_digit_templates()

# ---------------- 2. FUNCTIONS ----------------
def match_digit_from_image(roi_gray):
    if not digit_templates: return None
    roi = cv2.resize(roi_gray, (28, 28))
    _, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
    best_score, best_digit = -1, None
    for name, temp_img in digit_templates.items():
        res = cv2.matchTemplate(roi, temp_img, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score, best_digit = max_val, name
    return best_digit if best_score > 0.5 else None

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

# ---------------- 3. SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    active_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    st.divider()
    st.info(f"Templates Loaded: {len(digit_templates)}")

# ---------------- 4. MAIN SCAN UI ----------------
uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("ဖတ်နေသည်..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Image Enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)
            h, w = processed.shape
            
            col_w = w / active_cols
            grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]

            results = reader.readtext(processed)
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx, r_idx = int(cx // col_w), int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < active_cols:
                    x_min, y_min = int(min([p[0] for p in bbox])), int(min([p[1] for p in bbox]))
                    x_max, y_max = int(max([p[0] for p in bbox])), int(max([p[1] for p in bbox]))
                    roi = processed[max(0,y_min):min(h,y_max), max(0,x_min):min(w,x_max)]
                    
                    # Template Match check
                    matched = match_digit_from_image(roi)
                    final_text = matched if (matched and prob < 0.6) else text
                    
                    clean_txt = re.sub(r'[^0-9R]', '', str(final_text).upper())
                    if c_idx % 2 == 0: # ဂဏန်းတိုင်
                        grid_data[r_idx][c_idx] = clean_txt.zfill(3) if (clean_txt.isdigit() and len(clean_txt)<=3) else clean_txt
                    else: # ငွေပမာဏတိုင်
                        nums = re.findall(r'\d+', str(final_text))
                        grid_data[r_idx][c_idx] = nums[0] if nums else ""

            # Ditto Logic for Amount
            for c in range(1, active_cols, 2):
                last_val = ""
                for r in range(num_rows):
                    if grid_data[r][c] == "": grid_data[r][c] = last_val
                    else: last_val = grid_data[r][c]

            st.session_state['data_final'] = grid_data
            st.success("ဖတ်လို့ပြီးပါပြီ။")

# ---------------- 5. SHEET UPLOAD ----------------
if 'data_final' in st.session_state:
    st.subheader("Edit Data & Upload")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh1, sh2 = ss.get_worksheet(0), ss.get_worksheet(1)

            sh1.append_rows(edited_data)

            master_sum = {}
            for row in edited_data:
                for i in range(0, len(row)-1, 2):
                    if row[i] and row[i+1]:
                        res = process_bet_logic(row[i], row[i+1])
                        for k, v in res.items(): master_sum[k] = master_sum.get(k, 0) + v

            sh2.clear()
            final_summary = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["Number", "Total"]] + final_summary)
            st.success("✅ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ။")
        except Exception as e:
            st.error(f"Error: {e}")
