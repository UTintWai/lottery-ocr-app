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

# --- ပြင်ဆင်ချက်- ဒီ function ကို ထည့်ပေးဖို့ လိုအပ်ပါတယ် ---
def process_bet_logic(num_txt, amt_txt):
    """ ဂဏန်းနဲ့ အမှောင့်ကို ခွဲခြမ်းစိတ်ဖြာပြီး R (ပတ်သီး/အလှည့်) တွက်ချက်ခြင်း """
    num_clean = re.sub(r'[^0-9R.]', '', str(num_txt).upper())
    amt_clean = re.sub(r'[^0-9]', '', str(amt_txt))
    amt = int(amt_clean) if amt_clean else 0
    results = {}
    
    if 'R' in num_clean:
        base = num_clean.replace('R', '').replace('.', '')
        # ဂဏန်း ၃ လုံးဆိုလျှင် ၆ ပေါက်လှည့်မည်
        if len(base) == 3:
            perms = sorted(list(set([''.join(p) for p in permutations(base)])))
            split_amt = amt // len(perms)
            for p in perms: results[p] = split_amt
        else:
            results[base] = amt
    elif num_clean and num_clean != ".":
        results[num_clean] = amt
    return results

# ---------------- 2. IMPROVED RECOGNITION ----------------
def get_best_match(roi_gray):
    if not digit_templates: return ""
    _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        if w < 12 and h < 12: return "." # အစက်အပြောက်ဆိုလျှင် . ဟုယူမည်

    best_score = -1
    best_match = ""
    test_roi = cv2.resize(thresh, (28, 28))
    for name, temp in digit_templates.items():
        res = cv2.matchTemplate(test_roi, temp, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        if max_val > best_score:
            best_score = max_val
            best_match = name
    return best_match if best_score > 0.4 else ""

# ---------------- 3. MAIN UI ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    active_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("ဇယားကွက်တစ်ခုချင်းစီကို အသေးစိတ်ဖတ်နေပါသည်..."):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            row_h = h / num_rows
            col_w = w / active_cols
            grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]

            for r in range(num_rows):
                for c in range(active_cols):
                    y1, y2 = int(r * row_h), int((r + 1) * row_h)
                    x1, x2 = int(c * col_w), int((c + 1) * col_w)
                    roi = gray[y1+5:y2-5, x1+5:x2-5]
                    grid_data[r][c] = get_best_match(roi)

            st.session_state['data_final'] = grid_data
            st.success("✅ အားလုံးဖတ်ပြီးပါပြီ။")

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

            master_sum = {}
            for row in edited_data:
                # Number နှင့် Amount အတွဲလိုက်ကို summary လုပ်ရန်
                for i in range(0, len(row)-1, 2):
                    if row[i] and row[i+1]:
                        res = process_bet_logic(row[i], row[i+1])
                        for k, v in res.items():
                            master_sum[k] = master_sum.get(k, 0) + v

            sh2.clear()
            final_summary = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["Number", "Total"]] + final_summary)
            st.success("✅ Google Sheet သို့ အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ။")
        except Exception as e:
            st.error(f"Error: {e}")
