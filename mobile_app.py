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

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro 2026 Stable", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- BET LOGIC ---
def process_bet_logic(num_txt, amt_txt):
    num = re.sub(r'[^0-9R]', '', str(num_txt))
    amt_str = re.sub(r'[^0-9]', '', str(amt_txt))
    amt = int(amt_str) if amt_str else 0
    results = {}
    if 'R' in num:
        base = num.replace('R', '')
        if len(base) == 3:
            perms = sorted(list(set([''.join(p) for p in permutations(base)])))
            for p in perms: results[p] = amt // len(perms) if len(perms) > 0 else 0
        elif len(base) == 2:
            results[base] = amt; results[base[::-1]] = amt
    elif num:
        results[num] = amt
    return results

# --- SCAN FUNCTION ---
def scan_voucher_final(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    results = reader.readtext(gray, allowlist='0123456789R.*xX')
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        c = np.searchsorted(col_edges, cx) - 1
        r = np.searchsorted(row_edges, cy) - 1
        
        if 0 <= r < num_rows and 0 <= c < active_cols:
            t = text.upper().replace('X', '*')
            grid_data[r][c] = t
    return grid_data

# --- UI ---
st.title("🎯 Lottery Pro 2026")

with st.sidebar:
    st.header("⚙️ Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=2)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        with st.spinner("ဖတ်နေပါသည်..."):
            data = scan_voucher_final(img, a_cols, n_rows)
            st.session_state['sheet_data'] = data

# --- EDIT & SEND TO SHEET ---
if 'sheet_data' in st.session_state:
    st.subheader("📝 Edit Data")
    edited_df = st.data_editor(st.session_state['sheet_data'], use_container_width=True)
                   
    if st.button("🚀 Send to Google Sheet"):
        try:
            # 1. Secrets ကို ဖတ်ခြင်း
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            
            # 2. Credential Dictionary တည်ဆောက်ခြင်း
            creds_dict = {
                "type": info["type"],
                "project_id": info["project_id"],
                "private_key_id": info["private_key_id"],
                "private_key": info["private_key"].replace("\\n", "\n"),
                "client_email": info["client_email"],
                "client_id": info["client_id"],
                "auth_uri": info["auth_uri"],
                "token_uri": info["token_uri"],
                "auth_provider_x509_cert_url": info["auth_provider_x509_cert_url"],
                "client_x509_cert_url": info["client_x509_cert_url"]
            }
            
            # 3. Google Sheets ချိတ်ဆက်ခြင်း
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            
            # 4. Data ပို့ခြင်း (edited_df က List ဖြစ်နေရင် values.tolist() သုံးစရာမလိုပါ)
            # စာသားအလွတ်မဟုတ်တဲ့ Row တွေကိုပဲ စစ်ထုတ်ပြီး ပို့ပါမယ်
            clean_rows = [row for row in edited_df if any(str(cell).strip() for cell in row)]
            
            if clean_rows:
                sh1.append_rows(clean_rows)
                st.success("✅ Google Sheet ထဲ ရောက်သွားပါပြီဗျ!")
            else:
                st.warning("ပို့စရာ ဒေတာ မရှိပါဘူး။")

        except Exception as e:
            st.error(f"Error တက်နေပါတယ်: {str(e)}")
