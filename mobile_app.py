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
    # EasyOCR ကို CPU နဲ့ သုံးတဲ့အခါ မြန်အောင် Settings အချို့ ညှိထားပါတယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# --- SCAN FUNCTION ---
def scan_voucher_final(img, active_cols, num_rows):
    # ၁။ ပုံကို ၅၀% ချုံ့လိုက်ပါ (OCR ပိုမြန်သွားပါမယ်)
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5) 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape # ပုံရဲ့ အမြင့်နဲ့ အနံကို ယူပါတယ်

    # ၂။ detail=1 ထားမှ တည်နေရာ သိမှာပါ၊ ဒါပေမယ့် စာသားပဲ သီးသန့်ဖတ်ခိုင်းထားပါတယ်
    results = reader.readtext(gray, allowlist='0123456789R.*xX', detail=1) 
    
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
        with st.spinner("ဖတ်နေပါသည် (၁ မိနစ်ခန့် ကြာနိုင်သည်)..."):
            data = scan_voucher_final(img, a_cols, n_rows)
            st.session_state['sheet_data'] = data

# --- EDIT & SEND TO SHEET ---
if 'sheet_data' in st.session_state:
    st.subheader("📝 Edit Data")
    # data_editor ကနေ ရလာတဲ့ data ကို တိုက်ရိုက် သုံးပါမယ်
    edited_df = st.data_editor(st.session_state['sheet_data'], use_container_width=True)
                    
    if st.button("🚀 Send to Google Sheet"):
        try:
            # ၁။ Secrets ကို ဖတ်ခြင်း (dict() မသုံးပါနဲ့)
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"] 
            
            # ၂။ Credential dictionary ပြန်ဖွဲ့ခြင်း
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
            
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
            client = gspread.authorize(creds)
            
            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0)
            
            # ၃။ ဒေတာ သန့်စင်ပြီး ပို့ခြင်း
            # edited_df သည် list ဖြစ်နေသောကြောင့် values.tolist() သုံးရန်မလိုပါ
            clean_rows = [row for row in edited_df if any(str(cell).strip() for cell in row)]
            
            if clean_rows:
                sh1.append_rows(clean_rows)
                st.success("✅ Google Sheet ထဲ ရောက်သွားပါပြီဗျ!")
            else:
                st.warning("ပို့စရာ ဒေတာ မရှိပါဘူး။")

        except Exception as e:
            st.error(f"Error: {str(e)}")
