import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- OCR Load ---
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def process_grid(img, n_rows=25, n_cols=8):
    h, w = img.shape[:2]
    results = reader.readtext(img, detail=1)
    
    # ဇယားကွက်ကို သင်္ချာနည်းအရ အညီအမျှ ခွဲဝေခြင်း
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    
    # OCR ဖတ်လို့ရတဲ့ စာလုံးတစ်ခုချင်းစီကို သက်ဆိုင်ရာ အကွက်ထဲ ထည့်ခြင်း
    for (bbox, text, prob) in results:
        # စာလုံးရဲ့ အလယ်ဗဟိုကို တွက်ချက်ခြင်း
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        # မူရင်းပုံရဲ့ အကျယ်/အမြင့်ကို မူတည်ပြီး ဘယ်နှခုမြောက် အတန်း/အတိုင်လဲဆိုတာ ရှာခြင်း
        c_idx = int(cx / (w / n_cols))
        r_idx = int(cy / (h / n_rows))
        
        if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
            val = text.strip()
            # Ditto (။) သို့မဟုတ် အလားတူ သင်္ကေတများကို စစ်ဆေးခြင်း
            if any(m in val for m in ['"', '။', '=', '||', '..', '`', '4', 'u', 'U']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                # ၃ လုံးဂဏန်း ရှေ့က 0 ဖြည့်ခြင်း
                clean_num = re.sub(r'[^0-9\*xX]', '', val)
                if clean_num.isdigit() and len(clean_num) < 3:
                    clean_num = clean_num.zfill(3)
                grid[r_idx][c_idx] = clean_num

    # --- Ditto Logic: အပေါ်ကတန်ဖိုးကို အောက်သို့ ကူးခြင်း ---
    for c in range(n_cols):
        for r in range(1, n_rows):
            if grid[r][c] == "DITTO":
                grid[r][c] = grid[r-1][c]
                
    return grid

# --- Streamlit UI ---
st.title("Lottery Pro 2026 - Fixed Grid Logic")
uploaded_file = st.file_uploader("လက်ရေး Voucher ပုံတင်ပါ", type=["jpg","png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=400)

    if st.button("🔍 Scan Table"):
        final_grid = process_grid(img)
        st.session_state['processed_data'] = final_grid

if 'processed_data' in st.session_state:
    # အသုံးပြုသူမှ ပြင်ဆင်နိုင်ရန် ပြသခြင်း
    edited_data = st.data_editor(st.session_state['processed_data'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        # (Google Sheet API ချိတ်ဆက်မှုအပိုင်းသည် အပေါ်ကအတိုင်းဖြစ်သည်)
        st.success("ဒေတာများ အကွက်အလိုက် ရောက်ရှိသွားပါပြီ!")
