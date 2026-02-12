import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# --- 1. Page Configuration ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- 2. Google Sheets Connection Function ---
def get_gspread_client():
    try:
        if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            return gspread.authorize(creds)
    except Exception as e:
        st.error(f"Credentials Error: {e}")
    return None

# --- 3. OCR Model Loading ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- 4. Logic Functions (R-Permutation & Bet Parsing) ---
def get_r_list(num_str):
    """ဂဏန်း ၃ လုံး၏ ပတ်လည် ၅ လုံးကို ရှာခြင်း"""
    num_str = re.sub(r'\D', '', num_str)
    if len(num_str) != 3: return []
    all_perms = sorted(list(set([''.join(p) for p in permutations(num_str)])))
    if num_str in all_perms:
        all_perms.remove(num_str)
    return all_perms

def parse_bet_amount(amt_str):
    """1500*1000 logic: ရှေ့ဂဏန်းက အရင်း၊ နောက်ဂဏန်းက အပတ်"""
    amt_str = amt_str.replace(' ', '')
    if '*' in amt_str:
        parts = amt_str.split('*')
        main = int(re.sub(r'\D', '', parts[0])) if parts[0] else 0
        back = int(re.sub(r'\D', '', parts[1])) if len(parts) > 1 else 0
        return main, back
    # OCR က * ကို 0 လို့ မှားဖတ်ရင် ဥပမာ 150001000
    clean_amt = re.sub(r'\D', '', amt_str)
    return (int(clean_amt), 0) if clean_amt else (0, 0)

# --- 5. Sidebar Settings (ဒီမှာ အကုန်ပြန်ထည့်ထားပါတယ်) ---
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=50)
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"], index=3)
    num_cols_active = int(col_mode.split()[0])
    st.divider()
    st.write("Selected Rows:", num_rows)
    st.write("Selected Columns:", num_cols_active)

# --- 6. Main UI ---
st.title("🎰 Lottery Pro: Multi-Logic OCR")

uploaded_file = st.file_uploader("📥 လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(uploaded_file, caption="လက်ရှိတင်ထားသောပုံ", use_container_width=True)

    if st.button("🔍 စာဖတ်မည် (Start OCR)"):
        with st.spinner("စာသားများကို ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            results = reader.readtext(img)
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx = int((cx / w) * num_cols_active)
                r_idx = int((cy / h) * num_rows)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    txt = text.upper().strip().replace('S','5').replace('I','1').replace('Z','7').replace('O','0')
                    grid_data[r_idx][c_idx] = txt

            # --- Auto Formatting & Ditto Logic ---
            for c in range(num_cols_active):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    is_ditto = any(s in curr for s in ["\"", "||", "1", "U", "''", "။", "〃", "=", "-"])
                    
                    if (is_ditto or curr == "") and last_val != "":
                        grid_data[r][c] = last_val
                    elif curr != "":
                        if c % 2 == 0: # Number Column
                            clean_n = re.sub(r'[^0-9R*]', '', curr)
                            grid_data[r][c] = clean_n.zfill(3) if clean_n.isdigit() else clean_n
                            last_val = grid_data[r][c]
                        else: # Amount Column
                            grid_data[r][c] = curr.replace(' ', '')
                            last_val = grid_data[r][c]
            
            st.session_state['data_final'] = grid_data

# --- 7. Data Editor & Google Sheets Export ---
if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးရန် (အမှားပါက ဒီမှာပြင်ပါ)")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Google Sheets အားလုံးသို့ ပို့မည်"):
        client = get_gspread_client()
        if client:
            try:
                ss = client.open("LotteryData")
                sh1 = ss.get_worksheet(0)
                sh2 = ss.get_worksheet(1)
                try: sh3 = ss.get_worksheet(2)
                except: sh3 = ss.add_worksheet(title="Sheet3", rows="100", cols="5")

                # Sheet 1: Raw Save
                sh1.append_rows(edited_data)

                # Sheet 2 & 3 Processing
                master_sum = {}
                voucher_data = []

                for row in edited_data:
                    for i in range(0, 8, 2):
                        num = str(row[i]).strip()
                        bet_raw = str(row[i+1]).strip()
                        if num and bet_raw:
                            main_amt, r_total = parse_bet_amount(bet_raw)
                            
                            # ပင်မဂဏန်းပေါင်းခြင်း
                            master_sum[num] = master_sum.get(num, 0) + main_amt
                            
                            # R (ပတ်လည်) ခွဲဝေပေါင်းခြင်း
                            r_list = get_r_list(num)
                            if r_list and r_total > 0:
                                each_r = r_total // len(r_list)
                                for rn in r_list:
                                    master_sum[rn] = master_sum.get(rn, 0) + each_r
                            
                            # ၃၀၀၀ ကျော်ရင် Voucher ထုတ်ရန်
                            if (main_amt + r_total) > 3000:
                                voucher_data.append([num, (main_amt + r_total) - 3000, "Limit Over"])

                # Update Sheet 2 (Sorted Sum)
                sh2.clear()
                sorted_res = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
                sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + sorted_res)
                
                # Update Sheet 3 (Voucher)
                sh3.append_rows(voucher_data)

                st.success("🎉 Sheets 1, 2, 3 အားလုံး Update ဖြစ်သွားပါပြီ!")
            except Exception as e:
                st.error(f"Sheet Error: {e}")