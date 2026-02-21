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
    # ၈ တိုင်လုံး မိစေရန် Padding ထည့်ခြင်း
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
            # Ditto Logic
            if any(char in t for char in ['"', '။', '||', '..', '=']):
                grid_data[r][c] = "DITTO"
            else:
                grid_data[r][c] = t
    
    # Auto-fill Ditto
    for c in range(active_cols):
        for r in range(1, num_rows):
            if grid_data[r][c] == "DITTO":
                grid_data[r][c] = grid_data[r-1][c]
                
    return grid_data

# --- UI ---
st.title("🎯 Lottery Data Manager")

with st.sidebar:
    st.header("Settings")
    a_cols = st.selectbox("အတိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(img, use_container_width=True)

    if st.button("🔍 Scan ဖတ်မည်"):
        data = scan_voucher_final(img, a_cols, n_rows)
        # အကွက်လွတ်သော Row များကို ဖယ်ရှားခြင်း
        data = [row for row in data if any(cell != "" for cell in row)]
        st.session_state['sheet_data'] = data

if 'sheet_data' in st.session_state:
    # Scan ရလဒ်ကို ပြင်ဆင်နိုင်ရန် ပြခြင်း
    edited_df = st.data_editor(pd.DataFrame(st.session_state['sheet_data']), use_container_width=True)
    
    if st.button("🚀 Process & Save to Sheets"):
        try:
            # Google Sheets Connection
            info = st.secrets["GCP_SERVICE_ACCOUNT_FILE"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(info, ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"])
            client = gspread.authorize(creds)
            ss = client.open("LotteryData")
            
            # ၁။ ဒေတာများကို ဂဏန်းနှင့် ထိုးကြေးအဖြစ် ခွဲထုတ်ခြင်း (ဥပမာ- 123*500)
            parsed_data = []
            for row in edited_df.values.tolist():
                for cell in row:
                    if cell and '*' in str(cell):
                        pts = str(cell).split('*')
                        if len(pts) == 2 and pts[0].isdigit() and pts[1].isdigit():
                            parsed_data.append([pts[0], int(pts[1])])
            
            if not parsed_data:
                st.error("ဂဏန်း*ထိုးကြေး ပုံစံမျိုး မတွေ့ရပါ။ (ဥပမာ- 543*100)")
                st.stop()

            # ၂။ Sheet2: စုစည်းခြင်းနှင့် ငယ်စဉ်ကြီးလိုက်စီခြင်း
            sh2 = ss.worksheet("Sheet2")
            existing_sh2 = pd.DataFrame(sh2.get_all_records())
            new_df = pd.DataFrame(parsed_data, columns=['Number', 'Amount'])
            
            combined_df = pd.concat([existing_sh2, new_df], ignore_index=True)
            # ဂဏန်းတူလျှင် ထိုးကြေးပေါင်းမည်၊ ပြီးလျှင် ဂဏန်းအလိုက် စီမည်
            final_sh2 = combined_df.groupby('Number', as_index=False).sum().sort_values('Number')
            
            sh2.clear()
            sh2.update([final_sh2.columns.values.tolist()] + final_sh2.values.tolist())
            st.success("✅ Sheet2: စုစည်းစီရီပြီးပါပြီ။")

            # ၃။ Sheet3: ၃၀၀၀ ကျော်တာကို ဘောက်ချာထုတ်ခြင်း (အတန်း ၂၅ တန်းပုံသေ)
            sh3 = ss.worksheet("Sheet3")
            over_limit = final_sh2[final_sh2['Amount'] > 3000].copy()
            over_limit['Voucher'] = over_limit['Number'].astype(str) + "*" + (over_limit['Amount'] - 3000).astype(str)
            
            # ၂၅ တန်း ပုံသေ Format ယူခြင်း
            voucher_rows = [[v] for v in over_limit['Voucher'].tolist()]
            while len(voucher_rows) < 25: voucher_rows.append([""]) # ၂၅ တန်းပြည့်အောင်ဖြည့်
            
            sh3.clear()
            sh3.update("A1", [["Over 3000 Vouchers"]])
            sh3.update("A2", voucher_rows[:25])
            st.success("✅ Sheet3: ပိုလျံဘောက်ချာ (၂၅ တန်း) ထုတ်ပြီးပါပြီ။")

        except Exception as e:
            st.error(f"Error: {str(e)}")
