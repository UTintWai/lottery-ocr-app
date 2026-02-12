import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# --- Page Config ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- CSS for UI ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-color: #2e7d32; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Google Sheets Credentials ---
def get_gspread_client():
    if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
        secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
        secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
        return gspread.authorize(creds)
    return None

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- Sidebar Settings ---
with st.sidebar:
    st.title("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=50)
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"], index=3)
    num_cols_active = int(col_mode.split()[0])
    st.divider()
    st.info("၃၀၀၀ ကျော်သောဂဏန်းများကို Sheet 3 သို့ အလိုအလျောက် ခွဲထုတ်ပေးပါမည်။")

st.title("🎰 Lottery Pro: Advanced OCR")

# --- Upload & Processing ---
uploaded_file = st.file_uploader("📥 လက်ရေးမူပုံတင်ရန် (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    if st.button("🔍 စာဖတ်မည် (OCR Start)"):
        with st.spinner("စာဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            # 8 column grid for compatibility
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            results = reader.readtext(img)
            for (bbox, text, prob) in results:
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                c_idx = int((cx / w) * num_cols_active)
                r_idx = int((cy / h) * num_rows)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    txt = text.upper().strip().replace('S','5').replace('I','1').replace('Z','7').replace('O','0')
                    grid_data[r_idx][c_idx] = txt

            # --- Formatting & Auto-Correction Logic ---
            for c in range(num_cols_active):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    is_ditto = any(s in curr for s in ["\"", "||", "1", "U", "''", "။", "〃", "=", "-", "u"])
                    
                    # Error Correction for 36/24 reported
                    if c % 2 != 0: # Amount Column
                        if curr == "3" and "36" in last_val: curr = "36"
                        if curr == "2" and "24" in last_val: curr = "24"
                    
                    if (is_ditto or curr == "") and last_val != "":
                        grid_data[r][c] = last_val
                    elif curr != "":
                        if c % 2 == 0: # Number Column
                            clean_n = re.sub(r'\D', '', curr)
                            if clean_n:
                                formatted_n = clean_n[-3:].zfill(3)
                                grid_data[r][c] = formatted_n
                                last_val = formatted_n
                        else: # Amount Column
                            clean_a = re.sub(r'\D', '', curr)
                            grid_data[r][c] = clean_a
                            last_val = clean_a

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည် (1, 2, 3)"):
        client = get_gspread_client()
        if client:
            try:
                ss = client.open("LotteryData")
                sh1 = ss.get_worksheet(0) # Sheet 1
                sh2 = ss.get_worksheet(1) # Sheet 2
                try: sh3 = ss.get_worksheet(2)
                except: sh3 = ss.add_worksheet(title="Sheet3", rows="100", cols="5")

                # --- Sheet 1: Update ---
                sh1.append_rows(edited_data)

                # --- Sheet 2 & 3 Processing ---
                all_raw = sh1.get_all_values()
                master_sum = {}
                voucher_data = []

                for row in all_raw:
                    for i in range(0, 8, 2):
                        if i+1 < len(row):
                            num_val = str(row[i]).strip()
                            amt_val = str(row[i+1]).strip()
                            if num_val.isdigit() and amt_val.isdigit():
                                amt = int(amt_val)
                                # Sheet 2 Summing
                                master_sum[num_val] = master_sum.get(num_val, 0) + amt
                                # Sheet 3 Voucher (Over 3000)
                                if amt > 3000:
                                    voucher_data.append([num_val, amt - 3000, "ပိုငွေ"])

                # Update Sheet 2 (Sorted)
                sh2.clear()
                sorted_sums = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
                sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + sorted_sums)

                # Update Sheet 3 (Voucher)
                sh3.clear()
                sh3.append_rows([["ဂဏန်း", "ပိုငွေ (ဘောက်ချာ)", "မှတ်ချက်"]] + voucher_data)

                st.success("🎉 Sheets အားလုံး အောင်မြင်စွာ Update လုပ်ပြီးပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")