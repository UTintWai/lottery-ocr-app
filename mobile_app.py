import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- ၁။ CONFIG & FUNCTIONS ----------------
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

@st.cache_resource
def load_ocr():
    # GPU မရှိလျှင် False ထားပါ၊ စာလုံးအစိပ်ဆုံးဖတ်ရန် English တစ်ခုတည်းသုံးပါ
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3: return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X','*')
    results = {}
    try:
        if 'R' in clean_num:
            base = clean_num.replace('R','')
            perms = get_all_permutations(base)
            num_part = re.sub(r'\D','',amt_str)
            amt = int(num_part) if num_part else 0
            if perms and amt > 0:
                split = amt // len(perms)
                for p in perms: results[p] = split
        elif '*' in amt_str:
            parts = amt_str.split('*')
            if len(parts)==2:
                base_amt = int(re.sub(r'\D','',parts[0]))
                total_amt = int(re.sub(r'\D','',parts[1]))
                num_final = clean_num.zfill(3)
                results[num_final] = base_amt
                perms = [p for p in get_all_permutations(num_final) if p!=num_final]
                if perms:
                    split = (total_amt-base_amt)//len(perms)
                    for p in perms: results[p] = split
        else:
            num_part = re.sub(r'\D','',amt_str)
            amt = int(num_part) if num_part else 0
            num_final = clean_num.zfill(3) if (clean_num.isdigit() and len(clean_num)<=3) else clean_num
            if num_final: results[num_final] = amt
    except: pass
    return results

# ---------------- ၂။ SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုးသတ်မှတ်ရန်)", min_value=100, value=5000)
    num_rows = st.number_input("Rows (စာကြောင်းအရေအတွက်)", min_value=1, value=25)
    col_mode = st.selectbox("Columns (တိုင်အရေအတွက်)", ["2","4","6","8"], index=3) # Default 8
    num_cols_active = int(col_mode)

# ---------------- ၃။ OCR SCAN LOGIC ----------------
st.title("🎰 Lottery OCR 8-Column Stable")
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 စစ်ဆေးမည် (OCR Scan)"):
        with st.spinner(f"{num_cols_active} တိုင်စလုံးကို အနုစိတ် ဖတ်နေပါသည်..."):
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
                processed_img = clahe.apply(gray)
                
                h, w = img.shape[:2]
                grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]
                
                results = reader.readtext(processed_img, detail=1, contrast_ths=0.01, low_text=0.1, text_threshold=0.3)

                # Column Boundaries တွက်ချက်ခြင်း
                col_width = w / num_cols_active
                row_height = h / num_rows

                for (bbox, text, prob) in results:
                    cx = np.mean([p[0] for p in bbox])
                    cy = np.mean([p[1] for p in bbox])
                    
                    c_idx = int(cx // col_width)
                    r_idx = int(cy // row_height)

                    if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:
                        txt = text.upper().strip()
                        # Character Fixes
                        repls = {'O':'0','I':'1','S':'5','G':'6','Z':'7','B':'8','A':'4','T':'7','L':'1'}
                        for k, v in repls.items(): txt = txt.replace(k, v)
                        
                        if c_idx % 2 == 0: # ဂဏန်းတိုင်
                            txt = re.sub(r'[^0-9R]', '', txt)
                            if len(txt) == 2 and txt.isdigit(): txt = "0" + txt
                            elif len(txt) > 3 and 'R' not in txt: txt = txt[:3]
                        else: # ပမာဏတိုင်
                            txt = re.sub(r'[^0-9X*]', '', txt)
                        
                        grid_data[r_idx][c_idx] = txt

                # Ditto Logic
                for c in range(num_cols_active):
                    last_v = ""
                    for r in range(num_rows):
                        curr = str(grid_data[r][c]).strip()
                        if curr in ['"', "''", "4", "v", "V", "11", "ll", "LL", "-", "Y"] and last_v:
                            grid_data[r][c] = last_v
                        elif curr: last_v = curr
                
                st.session_state['data_final'] = grid_data
                st.rerun()
            except Exception as e:
                st.error(f"OCR Error: {str(e)}")

# ---------------- ၄။ SHEET UPLOAD ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Upload to Sheets"):
        try:
            # GCP Credentials
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n","\n")
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # ၁။ Sheet 1 (Raw) - တိုင်အရေအတွက်အတိုင်းပို့မည်
            sh1 = ss.get_worksheet(0)
            sh1.append_rows(edited_data)

            # ၂။ ပေါင်းခြင်း Logic
            master_sum = {}
            for row in edited_data:
                # ဇယားထဲရှိ အကွက်တိုင်းကို ၂ ကွက်တွဲစီ စစ်မည်
                for i in range(0, len(row)-1, 2):
                    n, a = str(row[i]).strip(), str(row[i+1]).strip()
                    if n and a:
                        bet_res = process_bet_logic(n, a)
                        for k, v in bet_res.items():
                            master_sum[k] = master_sum.get(k, 0) + v

            # ၃။ Sheet 2 (Total)
            sh2 = ss.get_worksheet(1)
            sh2.clear()
            sh2.append_rows([["Number", "Total"]] + [[k, v] for k, v in sorted(master_sum.items())])

            # ၄။ Sheet 3 (Excess)
            sh3 = ss.get_worksheet(2)
            sh3.clear()
            excess_rows = [[k, v - bet_limit] for k, v in master_sum.items() if v > bet_limit]
            if excess_rows:
                sh3.append_rows([["ဂဏန်း", "ပိုလျှံငွေ"]] + sorted(excess_rows))
                st.success(f"✅ Sheet 1, 2, 3 အားလုံး အောင်မြင်စွာ ပို့ပြီးပါပြီ။")
            else:
                st.success("✅ Sheet 1, 2 ပို့ပြီးပါပြီ။ (ပိုလျှံဂဏန်း မရှိပါ)")

        except Exception as e:
            st.error(f"Sheet Error: {str(e)}")