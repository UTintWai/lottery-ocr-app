import streamlit as st
import numpy as np
import easyocr
import cv2
import re

st.set_page_config(page_title="Lottery Pro 2026 Ultimate", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def clean_ocr_text(txt):
    # စာလုံးအမှားများကို ဂဏန်းသို့ ပြောင်းခြင်း
    txt = txt.upper().strip()
    repls = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'7','T':'7','Q':'0','D':'0'}
    for k,v in repls.items():
        txt = txt.replace(k,v)
    return txt

def advanced_processing(img):
    # ၁။ Gray ပြောင်းပြီး Contrast မြှင့်တင်ခြင်း
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ၂။ Noise ဖယ်ရှားခြင်း (Denoising) - လက်ရေးများ ပိုကြည်လင်စေရန်
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    
    # ၃။ Adaptive Thresholding (လက်ရေးကို အနက်ရောင်ပြောင်းပြီး နောက်ခံကို အဖြူသားဖြစ်စေခြင်း)
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # ၄။ Sharpening (စာလုံးစွန်းများကို ပိုထက်မြက်စေရန်)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    return sharpened

# --- UI ---
with st.sidebar:
    st.header("⚙ Grid Settings")
    n_rows = st.number_input("Rows (အတန်း)", min_value=1, value=25)
    a_cols = st.selectbox("Columns (အတိုင်)", [2, 4, 6, 8], index=3)

uploaded_file = st.file_uploader("Upload Voucher Image", type=["jpg","jpeg","png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, caption="Original Image", use_container_width=True)

    if st.button("🔍 Scan & Fix Table"):
        with st.spinner("လက်ရေးများကို သန့်စင်ပြီး ဖတ်နေပါသည်..."):
            # Image Processing
            processed_img = advanced_processing(img)
            st.image(processed_img, caption="Cleaned Image for OCR", width=400)
            
            h, w = processed_img.shape
            grid_data = [["" for _ in range(a_cols)] for _ in range(n_rows)]
            
            # OCR ဖတ်ခြင်း (detail=1 ပါမှ တည်နေရာသိရမည်)
            results = reader.readtext(processed_img, detail=1)
            
            # ဇယားကွက် ပိုင်းခြားခြင်း
            col_edges = np.linspace(0, w, a_cols + 1)
            row_edges = np.linspace(0, h, n_rows + 1)

            for (bbox, text, prob) in results:
                # စာလုံး၏ အလယ်ဗဟိုကို ယူခြင်း
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                # မည်သည့် Column/Row ထဲရောက်သည်ကို တွက်ခြင်း
                c_idx = np.searchsorted(col_edges, cx) - 1
                r_idx = np.searchsorted(row_edges, cy) - 1
                
                if 0 <= r_idx < n_rows and 0 <= c_idx < a_cols:
                    txt = clean_ocr_text(text)
                    # ဂဏန်းနှင့် special characters များယူခြင်း (ဥပမာ 123*500)
                    match = re.search(r'[\d\*\.xX]+', txt)
                    if match:
                        clean_val = match.group().replace('X', '*').replace('x', '*')
                        if grid_data[r_idx][c_idx] == "":
                            grid_data[r_idx][c_idx] = clean_val
                        else:
                            grid_data[r_idx][c_idx] += f" {clean_val}"

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 ရလဒ် (လိုအပ်ပါက ပြင်ဆင်နိုင်သည်)")
    # Data Editor ကိုသုံးခြင်းဖြင့် အမှားများကို ချက်ချင်းပြင်နိုင်သည်
    st.data_editor(st.session_state['data_final'], use_container_width=True)
