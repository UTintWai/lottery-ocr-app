import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# --- Configuration ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- Betting Logic: R-Permutation Split ---
def get_all_permutations(num_str):
    """ဂဏန်းတစ်ခု၏ ဖြစ်နိုင်သမျှ ပတ်လည် အားလုံးကို ရှာခြင်း (မူရင်းအပါအဝင်)"""
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3: return [num_only] if num_only else []
    perms = sorted(list(set([''.join(p) for p in permutations(num_only)])))
    return perms

def process_bet_logic(num_txt, amt_txt):
    """
    R ပါသော ဂဏန်းများနှင့် ထိုးကြေးများကို ခွဲဝေခြင်း
    ဥပမာ: 267R, 360 -> {'267':60, '276':60, ...}
    """
    clean_num = re.sub(r'[^0-9R]', '', num_txt.upper())
    clean_amt = re.sub(r'\D', '', amt_txt)
    amt = int(clean_amt) if clean_amt else 0
    
    results = {}
    
    if 'R' in clean_num:
        base_num = clean_num.replace('R', '')
        perms = get_all_permutations(base_num)
        if perms and amt > 0:
            split_amt = amt // len(perms)
            for p in perms:
                results[p] = split_amt
    else:
        # R မပါရင် ပုံမှန်အတိုင်းပဲ ပေါင်းမယ်
        if clean_num.isdigit():
            results[clean_num.zfill(3)] = amt
            
    return results

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=50)
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"], index=3)
    num_cols = int(col_mode.split()[0])
    st.divider()
    st.info("Logic: 267R-360 ဆိုလျှင် ၆ ကွက်အား ၆၀ စီ ခွဲဝေပေးမည်။")

st.title("🎰 Lottery OCR: R-System Enabled")

uploaded_file = st.file_uploader("📥 လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 စာဖတ်မည်"):
        with st.spinner("ဖတ်နေပါသည်..."):
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            results = reader.readtext(img)
            
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx, r_idx = int((cx/w)*num_cols), int((cy/h)*num_rows)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    txt = text.upper().strip().replace('S','5').replace('I','1').replace('Z','7').replace('G','6')
                    # ဂဏန်းတိုင်ဖြစ်ပါက R ကလွဲပြီး အင်္ဂလိပ်စာလုံးများ ဖယ်ရှားမည်
                    if c_idx % 2 == 0:
                        txt = re.sub(r'[^0-9R]', '', txt)
                    grid_data[r_idx][c_idx] = txt

            # Formatting & Ditto Logic
            for c in range(num_cols):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if curr in ["", "။", "\"", "||", "="] and last_val != "":
                        grid_data[r][c] = last_val
                    elif curr != "":
                        last_val = curr
            
            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြင်ဆင်ရန် (ဥပမာ- 267R)")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Google Sheets သို့ ခွဲဝေပို့ဆောင်မည်"):
        # (Google Sheets Connection Logic here...)
        try:
            client = get_gspread_client() # type: ignore
            ss = client.open("LotteryData")
            sh1, sh2, sh3 = ss.get_worksheet(0), ss.get_worksheet(1), ss.get_worksheet(2)
            
            sh1.append_rows(edited_data) # Raw save
            
            master_sum = {}
            voucher_list = []

            for row in edited_data:
                for i in range(0, 8, 2):
                    n_txt, a_txt = str(row[i]), str(row[i+1])
                    if n_txt and a_txt:
                        # R logic ဖြင့် ခွဲဝေခြင်း
                        bet_results = process_bet_logic(n_txt, a_txt)
                        for g, val in bet_results.items():
                            master_sum[g] = master_sum.get(g, 0) + val
                        
                        # Sheet 3 ပိုငွေအတွက် (မူရင်းထိုးကြေး ၃၀၀၀ ကျော်လျှင်)
                        amt_num = int(re.sub(r'\D', '', a_txt)) if re.sub(r'\D', '', a_txt) else 0
                        if amt_num > 3000:
                            voucher_list.append([n_txt, amt_num - 3000, "ပိုငွေ"])

            # Sheet 2 Update (Sorted)
            sh2.clear()
            final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + final_list)
            
            # Sheet 3 Update
            if voucher_list: sh3.append_rows(voucher_list)
            
            st.success("🎉 R-System ဖြင့် အောင်မြင်စွာ ပေါင်းစည်းပြီးပါပြီ!")
        except Exception as e:
            st.error(f"Error: {e}")