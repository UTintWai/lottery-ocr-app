import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026 Stable", layout="wide")

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

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("Rows", min_value=1, value=25)
    col_mode = st.selectbox("Columns", ["2","4","6","8"], index=3)
    num_cols_active = int(col_mode)

st.title("🎰 Lottery OCR Ultra Stable 2026")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):
        with st.spinner(f"Scanning {num_cols_active} columns..."):

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)

            h, w = processed.shape
            col_width = w / num_cols_active   # ✅ အမှန်
            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

results = reader.readtext(
    processed,
    detail=1,
    paragraph=False,
    width_ths=0.7,
    height_ths=0.7
)

for (bbox, text, prob) in results:
    if prob < 0.40:
        continue
    cx = np.mean([p[0] for p in bbox])
    cy = np.mean([p[1] for p in bbox])

    c_idx = min(int(cx / col_width), num_cols_active-1)
    r_idx = int((cy / h) * num_rows)

    if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:
        txt = text.upper().strip()
        repls = {'S':'5','G':'6','I':'1','Z':'7','B':'8','O':'0','L':'1','T':'7','Q':'0','D':'0'}
        for k,v in repls.items():
            txt = txt.replace(k,v)
        if c_idx % 2 == 0:
            txt = re.sub(r'[^0-9R]', '', txt)
        grid_data[r_idx][c_idx] = txt

        # Ditto + Number/Amount Logic
        for c in range(num_cols_active):
            last_val = ""
            for r in range(num_rows):
                curr = str(grid_data[r][c]).strip().upper()
                if c % 2 == 0:  # number columns
                    curr = re.sub(r'[^0-9R]', '', curr)
                    if curr.isdigit():
                        curr = curr[-3:].zfill(3)
                        if last_val and curr == last_val:
                            grid_data[r][c] = str(int(last_val) + int(curr))
                        else:
                            grid_data[r][c] = curr
                            if curr:
                                last_val = curr
                    else:  # amount columns
                        nums = re.findall(r'\d+', curr)
                        curr = max(nums, key=lambda x: int(x)) if nums else ""
                        if (curr == "" or (curr.isdigit() and len(curr) <= 2)) and last_val:
                            grid_data[r][c] = last_val
                        else:
                            grid_data[r][c] = curr
                            if curr:
                                last_val = curr

            st.session_state['data_final'] = grid_data

# ---------------- GOOGLE SHEET ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Upload to Google Sheet"):
        try:
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n","\n")
            scope = ["https://spreadsheets.google.com/feeds","https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)

            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            sh2 = ss.get_worksheet(1)

            sh1.append_rows(edited_data)

            master_sum = {}
            for row in edited_data:
                for i in range(0, num_cols_active, 2):
                    n_txt = str(row[i]).strip()
                    a_txt = str(row[i+1]).strip()
                    if n_txt and a_txt:
                        bet_res = process_bet_logic(n_txt, a_txt)
                        for g,val in bet_res.items():
                            master_sum[g] = master_sum.get(g,0)+val

            sh2.clear()
            final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["Number","Total"]] + final_list)

            st.success("✅ Uploaded Successfully!")
        except Exception as e:
            st.error(f"Error: {str(e)}")
