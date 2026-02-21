import streamlit as st
import numpy as np
import easyocr
import cv2
import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro 2026 Ultimate", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def scan_voucher_final(img, active_cols, num_rows):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.copyMakeBorder(gray, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    h, w = gray.shape
    results = reader.readtext(gray, allowlist='0123456789R.*xX" ', detail=1) 
    
    grid_data = [["" for _ in range(active_cols)] for _ in range(num_rows)]
    col_edges = np.linspace(0, w, active_cols + 1)
    row_edges = np.linspace(0, h, num_rows + 1)

    for (bbox, text, prob) in results:
        cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
        c, r = np.searchsorted(col_edges, cx) - 1, np.searchsorted(row_edges, cy) - 1
        if 0 <= r < num_rows and 0 <= c < active_cols:
            t = text.upper().replace('X', '*').strip()
            is_ditto = any(char in t for char in ['"', '။', '||', '..', '='])
            if is_ditto: grid_data[r][c] = "DITTO"
            elif grid_data[r][c] == "": grid_data[r][c] = t
            else: grid_data[r][c] += f" {t}"
    
    for c in range(active_cols):
        for r in range(1, num_rows):
            if grid_data[r][c] == "DITTO": grid_data[r][c] = grid_data[r-1][c]
                
    return grid_data

# --- APP UI ---
st.title("🎯 Lottery System (Sorted & Voucher Mode)")

with st.sidebar:
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan စတင်မည်"):
        data = scan_voucher_final(img, a_cols, n_rows)
        st.session_state['sheet_data'] = data

if 'sheet_data' in st.session_state:
    edited_data = st.data_editor(st.session_state['sheet_data'], use_container_width=True)
                    
    if st.button("🚀 Process & Send Data"):
        try:
            # Google Sheets Connection
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # ၁။ ဒေတာများကို ဂဏန်းနှင့် ထိုးကြေးအဖြစ် ခွဲထုတ်ခြင်း
            all_entries = []
            for row in edited_data:
                for cell in row:
                    if '*' in str(cell):
                        parts = cell.split('*')
                        if len(parts) == 2:
                            num, amt = parts[0].strip(), parts[1].strip()
                            if num.isdigit() and amt.isdigit():
                                all_entries.append({'Number': num, 'Amount': int(amt)})

            if not all_entries:
                st.warning("ပို့ရန် ဒေတာ မရှိပါ။ (ဂဏန်း*ထိုးကြေး ပုံစံရှိရပါမည်)")
                st.stop()

            df_new = pd.DataFrame(all_entries)

            # ၂။ Sheet2 အတွက် ဒေတာစုစည်းခြင်း (Grouping & Sorting)
            sh2 = ss.worksheet("Sheet2")
            existing_sh2 = pd.DataFrame(sh2.get_all_records())
            
            df_combined = pd.concat([existing_sh2, df_new], ignore_index=True)
            df_final_sh2 = df_combined.groupby('Number', as_index=False).sum()
            df_final_sh2 = df_final_sh2.sort_values(by='Number') # ငယ်စဉ်ကြီးလိုက် စီခြင်း

            sh2.clear()
            sh2.update([df_final_sh2.columns.values.tolist()] + df_final_sh2.values.tolist())
            st.success("✅ Sheet2: ငယ်စဉ်ကြီးလိုက် စုစည်းပြီးပါပြီ!")

            # ၃။ Sheet3 အတွက် ၃၀၀၀ ကျော်တာများကို Voucher ပုံစံလုပ်ခြင်း
            sh3 = ss.worksheet("Sheet3")
            df_over = df_final_sh2[df_final_sh2['Amount'] > 3000].copy()
            df_over['Over_Amount'] = df_over['Amount'] - 3000
            
            # Voucher Format (Number*OverAmount)
            voucher_list = [f"{r['Number']}*{r['Over_Amount']}" for _, r in df_over.iterrows()]
            
            # ၂၅ တန်းပုံသေ သတ်မှတ်ခြင်း
            final_voucher_rows = [[v] if i < len(voucher_list) else [""] for i in range(25)]
            
            sh3.clear()
            sh3.update("A1", [["Voucher (Over 3000)"]])
            sh3.update("A2", final_voucher_rows)
            st.success("✅ Sheet3: ပိုလျံထိုးကြေး ဘောက်ချာ ထုတ်ပြီးပါပြီ!")

        except Exception as e:
            st.error(f"Error: {str(e)}")
