import streamlit as st
import numpy as np
import easyocr
import cv2
import re

# --- OCR Load ---
@st.cache_resource
def load_ocr():
    # 'en' အပြင် တခြားနံပါတ်တွေပါ ဖတ်နိုင်အောင် recognition နည်းနည်းမြှင့်ထားတယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def enhance_image(img):
    # ပုံကို ပိုကြည်အောင် Gray ပြောင်းပြီး Contrast မြှင့်မယ်
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Noise လျှော့ချခြင်း
    gray = cv2.medianBlur(gray, 3)
    # Adaptive Threshold သုံးပြီး စာလုံးကို ပေါ်လွင်အောင်လုပ်ခြင်း
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return thresh

def process_lottery(img, n_rows, n_cols):
    h, w = img.shape[:2]
    # ပုံကို ကြည်အောင် အရင်လုပ်မယ်
    processed_img = enhance_image(img)
    
    # OCR ဖတ်မယ်
    results = reader.readtext(processed_img)
    
    # Grid table ဆောက်မယ်
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    
    # Column တစ်ခုချင်းစီရဲ့ width ကို တွက်မယ်
    col_edges = np.linspace(0, w, n_cols + 1)
    row_edges = np.linspace(0, h, n_rows + 1)

    for (bbox, text, prob) in results:
        # စာလုံးရဲ့ အလယ်မှတ်ကို ယူမယ်
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        
        # ဘယ် Column/Row ထဲမှာ ရှိလဲဆိုတာ ရှာမယ်
        c_idx = np.searchsorted(col_edges, cx) - 1
        r_idx = np.searchsorted(row_edges, cy) - 1
        
        if 0 <= r_idx < n_rows and 0 <= c_idx < n_cols:
            clean_text = text.strip()
            
            # Ditto သတ်မှတ်ချက် (သင်သုံးထားတဲ့ Logic အတိုင်း)
            if any(m in clean_text for m in ['"', '။', '=', '||', '..', '`']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                # ဂဏန်းသက်သက်ပဲ ယူပြီး ၃ လုံးဖြည့်မယ်
                nums = re.sub(r'[^0-9]', '', clean_text)
                if nums:
                    grid[r_idx][c_idx] = nums.zfill(3)

    # Ditto Fill-down Logic
    for c in range(n_cols):
        for r in range(1, n_rows):
            if grid[r][c] == "DITTO" and grid[r-1][c] != "":
                grid[r][c] = grid[r-1][c]
                
    return grid

# --- UI Layout ---
st.title("🎯 Lottery Pro 2026 (Fix Version)")

with st.sidebar:
    mode = st.radio("တိုင်အရေအတွက် ရွေးပါ", ["၆ တိုင်", "၈ တိုင်"])
    a_cols = 6 if mode == "၆ တိုင်" else 8
    n_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("Voucher ပုံတင်ပါ", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="တင်ထားသောပုံ", use_column_width=True)

    if st.button("🔍 Scan အခုလုပ်မယ်"):
        with st.spinner(f"{a_cols} တိုင်အတွက် ဖတ်နေသည်..."):
            data = process_lottery(img, n_rows, a_cols)
            st.session_state['data'] = data

if 'data' in st.session_state:
    st.write("### ရလဒ် (ပြင်ဆင်နိုင်သည်)")
    edited_df = st.data_editor(st.session_state['data'])
