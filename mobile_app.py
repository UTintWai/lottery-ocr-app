import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026 Auto Detect", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- PERMUTATION ----------------
def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3:
        return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

# ---------------- BET LOGIC ----------------
def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X', '*')
    results = {}

    try:
        if 'R' in clean_num:
            base = clean_num.replace('R', '')
            perms = get_all_permutations(base)
            amt = int(re.sub(r'\D', '', amt_str)) if re.sub(r'\D', '', amt_str) else 0
            if perms and amt > 0:
                split = amt // len(perms)
                for p in perms:
                    results[p] = split

        elif '*' in amt_str:
            parts = amt_str.split('*')
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                base_amt = int(parts[0])
                total_amt = int(parts[1])
                num_final = clean_num.zfill(3)
                results[num_final] = base_amt
                perms = [p for p in get_all_permutations(num_final) if p != num_final]
                if perms:
                    split = (total_amt - base_amt) // len(perms)
                    for p in perms:
                        results[p] = split
        else:
            amt = int(re.sub(r'\D', '', amt_str)) if re.sub(r'\D', '', amt_str) else 0
            num_final = clean_num.zfill(3) if clean_num.isdigit() else clean_num
            if num_final:
                results[num_final] = amt
    except:
        pass

    return results
uploaded_file = st.file_uploader("Upload Voucher Image", type=["jpg","jpeg","png"])
# ---------------- OCR NORMALIZE ----------------
def clean_ocr_text(txt):
    txt = txt.upper().strip()

    # Remove .png / PNG / .PNG etc
    txt = re.sub(r'\.?\s*PNG', '', txt)

    # Fix common OCR mistakes
    repls = {
        'O':'0','I':'1','L':'1','S':'5',
        'B':'8','G':'6','Z':'7','T':'7',
        'Q':'0','D':'0'
    }

    for k,v in repls.items():
        txt = txt.replace(k,v)

    return txt
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("Rows", min_value=1, value=25)
    col_mode = st.selectbox("Columns (Manual Override)", ["Auto Detect","2","4","6","8"], index=0)

...

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):
        with st.spinner("Scanning & Auto Detecting Columns..."):

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)

            h, w = processed.shape

            # AUTO DETECT
            proj = np.sum(processed, axis=0)
            threshold = np.percentile(proj, 75)
            text_mask = proj < threshold
            col_indices = np.where(text_mask)[0]

            clusters = []
            if len(col_indices) > 0:
                current_cluster = [col_indices[0]]
                gap_limit = w // 20
                for idx in col_indices[1:]:
                    if idx - current_cluster[-1] < gap_limit:
                        current_cluster.append(idx)
                    else:
                        clusters.append(current_cluster)
                        current_cluster = [idx]
                clusters.append(current_cluster)

            num_cols_detected = len(clusters)
            if num_cols_detected < 6:
                num_cols_detected = 6
            elif num_cols_detected > 8:
                num_cols_detected = 8

            # Manual override
            if col_mode != "Auto Detect":
                num_cols_active = int(col_mode)
            else:
                num_cols_active = num_cols_detected

            col_width = w / num_cols_active
            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

            st.success(f"🔎 Columns Used: {num_cols_active}")

            # OCR fixes dictionary
            repls = {
                'S':'5','G':'6','[':'1','Z':'7','9.png':'9','222.png':'222',
                'B':'8','O':'0','L':'1','T':'7','5.png':'5','7.png':'7','r.png':'R',
                'Q':'0','D':'0','ဒ':'3','lZO':'120','1.png':'1','I':'1','l':'1',
                '|':'1','!':'1',']':'1',                                                                                           '3.png':'3',
                'IZO':'120','12O':'120','[ZO':'120',']ZO':'120','120.png':'120',
                'IZO':'120','12O':'120','LZO':'120','I2O':'120', '1Z0':'120',
                '120O':'120','l2O':'120','IZ0':'120'
            }
            txt = text.upper().strip()  # type: ignore
            for k,v in repls.items():
                txt = txt.replace(k,v)

            # number column normalize
        if c_idx % 2 == 0: # type: ignore
            txt = re.sub(r'[^0-9R]', '', txt)
            if txt.isdigit():
                txt = txt.zfill(3)

# OCR READ
results = reader.readtext(processed, detail=1, paragraph=False)

for (bbox, text, prob) in results:
    if prob < 0.40:
        continue

    cx = np.mean([p[0] for p in bbox])
    cy = np.mean([p[1] for p in bbox])

    c_idx = int(cx / col_width)
    r_idx = int((cy / h) * num_rows)

    if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:

        # Normalize OCR text
        txt = text.upper().strip()
        for k,v in repls.items():
            txt = txt.replace(k,v)

        # NUMBER COLUMN
        if c_idx % 2 == 0:
            nums = re.findall(r'\d+', txt)
            if nums:
                txt = nums[0].zfill(3)
            else:
                txt = ""

        # AMOUNT COLUMN
        else:
            nums = re.findall(r'\d+', txt)
            txt = max(nums, key=lambda x: int(x)) if nums else ""

        grid_data[r_idx][c_idx] = txt

            # Ditto Logic
        for c in range(num_cols_active):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if c % 2 == 0:  # number column
                        curr = re.sub(r'[^0-9R]', '', curr)
                        if curr.isdigit():
                            curr = curr.zfill(3)
                            # prevent duplicate above/below
                            if r > 0 and curr == grid_data[r-1][c]:
                                curr = ""  # clear duplicate
                                grid_data[r][c] = curr if curr else last_val
                                if curr:
                                    last_val = curr
                  
                    else:  # amount column
                        nums = re.findall(r'\d+', curr)
                        curr = max(nums, key=lambda x: int(x)) if nums else ""
                    grid_data[r][c] = curr if curr else last_val
                    if curr:
                        last_val = curr

        st.session_state['data_final'] = grid_data
        st.session_state['num_cols'] = num_cols_active

# ---------------- GOOGLE SHEET ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Upload to Google Sheet"):
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n","\n")

            scope = ["https://spreadsheets.google.com/feeds",
                     "https://www.googleapis.com/auth/drive"]

            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)

            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            sh2 = ss.get_worksheet(1)

            sh1.append_rows(edited_data)

            master_sum = {}
            num_cols_active = st.session_state['num_cols']

            for row in edited_data:
                for i in range(0, len(row)-1, 2):
                    n_txt = str(row[i]).strip()
                    a_txt = str(row[i+1]).strip()

                    if n_txt and a_txt:
                        bet_res = process_bet_logic(n_txt, a_txt)
                        for g,val in bet_res.items():
                            master_sum[g] = master_sum.get(g,0) + val

            sh2.clear()
            final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["Number","Total"]] + final_list)

            st.success("✅ Uploaded Successfully!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
