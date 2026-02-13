import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# --- ၁။ Page Configuration ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- ၂။ OCR Model Loading ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- ၃။ Betting Logic (R-System) ---
def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3: return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X','*')
    results = {}

    # Case 1: R-system
    if 'R' in clean_num:
        base_num = clean_num.replace('R', '')
        perms = get_all_permutations(base_num)
        amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
        if perms and amt > 0:
            split_amt = amt // len(perms)
            for p in perms:
                results[p] = split_amt

    # Case 2: Multiplier expression (e.g. 1500*1000, 50*50)
    elif '*' in amt_str:
        parts = amt_str.split('*')
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            base_amt = int(parts[0])
            total_amt = int(parts[1])
            # base number
            num_final = clean_num.zfill(3)
            results[num_final] = base_amt
            # remaining permutations
            perms = [p for p in get_all_permutations(num_final) if p != num_final]
            if perms:
                split_amt = (total_amt - base_amt) // len(perms)
                for p in perms:
                    results[p] = split_amt

    # Case 3: Normal digit amount
    else:
        amt = int(re.sub(r'\D','',amt_str)) if re.sub(r'\D','',amt_str) else 0
        num_final = clean_num.zfill(3) if (clean_num.isdigit() and len(clean_num) <= 3) else clean_num
        if num_final:
            results[num_final] = amt

    return results

# --- ၄။ Sidebar Settings ---
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"], index=3)
    num_cols_active = int(col_mode.split()[0])
    st.divider()
    st.info("Logic: 267R-360 ဆိုလျှင် ပတ်လည် ၆ ကွက်ကို ၆၀ စီ ခွဲဝေပေးပါမည်။")

# --- ၅။ Main UI ---
st.title("🎰 Lottery OCR (Fixed Sheet Upload)")

uploaded_file = st.file_uploader("📥 လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 စစ်ဆေးမည် (OCR Scan)"):
     with st.spinner("ဖတ်နေပါသည်..."):
        h, w = img.shape[:2]
        grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
        results = reader.readtext(img)

        for (bbox, text, prob) in results:
            cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
            c_idx = int((cx / w) * num_cols_active)
            r_idx = int((cy / h) * num_rows)

            if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                txt = text.upper().strip().replace('S','5').replace('I','1').replace('Z','7').replace('G','6')
                grid_data[r_idx][c_idx] = txt

        # --- OCR Result Formatting & Strict Filtering ---
        for c in range(num_cols_active):
            last_val = ""
            for r in range(num_rows):
                curr = str(grid_data[r][c]).strip().upper()

                if c % 2 == 0:  # number columns
                    curr = curr.replace('S','5').replace('I','1').replace('Z','7').replace('G','6')
                    curr = re.sub(r'[^0-9R]', '', curr)
                    if curr:
                        if curr.isdigit():
                            m = re.search(r'(\d{3})$', curr)
                            if m:
                                curr = m.group(1)
                            else:
                                curr = curr[-3:].zfill(3)
                else:  # amount columns
                    nums = re.findall(r'\d+', curr)
                    curr = max(nums, key=lambda x: int(x)) if nums else ""

                # Ditto logic
                if (curr == "" or (curr.isdigit() and len(curr) <= 2)) and last_val:
                    grid_data[r][c] = last_val
                else:
                    grid_data[r][c] = curr
                    if curr:
                        last_val = curr

        st.session_state['data_final'] = grid_data

# --- ၆။ Google Sheets Upload Section ---
if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး Google Sheet သို့ ပို့ရန်")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Google Sheet သို့ အကုန်ပို့မည်"):
        try:
            # Credentials Connection
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)
            
            # Open Spreadsheet
            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0) # Raw Data
            sh2 = ss.get_worksheet(1) # Sum Data
            try: sh3 = ss.get_worksheet(2)
            except: sh3 = ss.add_worksheet(title="Sheet3", rows="100", cols="5")

            # --- Sheet 1: Append Edited Data ---
            sh1.append_rows(edited_data)

            # --- Sheet 2 & 3 Processing ---
            master_sum = {}
            voucher_list = []

            for row in edited_data:
                for i in range(0, 8, 2):
                    n_txt, a_txt = str(row[i]).strip(), str(row[i+1]).strip()
                    if n_txt and a_txt:
                        # R Logic ခွဲဝေခြင်း
                        bet_res = process_bet_logic(n_txt, a_txt)
                        for g, val in bet_res.items():
                            master_sum[g] = master_sum.get(g, 0) + val
                        
                        # Sheet 3 (ပိုငွေ ၃၀၀၀ ကျော်လျှင်)
                        amt_num = int(re.sub(r'\D', '', a_txt)) if re.sub(r'\D', '', a_txt) else 0
                        if amt_num > 3000:
                            voucher_list.append([n_txt, amt_num - 3000, "ပိုငွေ"])

            # --- Sheet 2: Clear and Update with Sorted Sum ---
            sh2.clear()
            final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + final_list)
            
            # --- Sheet 3: Update ---
            if voucher_list:
                sh3.append_rows(voucher_list)
            
            st.success("🎉 Sheet 1, 2 နှင့် 3 သို့ အချက်အလက်များ အောင်မြင်စွာ ပို့ပြီးပါပြီ!")
            
        except Exception as e:
            st.error(f"❌ Error: {e}")
            st.write("အကယ်၍ Spreadsheet အမည် မမှန်ကန်ပါက 'LotteryData' ဖြစ်အောင် ပြင်ပေးပါဗျ။")