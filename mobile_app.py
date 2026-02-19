import streamlit as st
import numpy as np
import easyocr
import cv2
import re

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- CLEAN OCR ----------------
def clean_ocr_text(txt):
    txt = txt.upper().strip()
    repls = {
        'O':'0','I':'1','L':'1','S':'5',
        'B':'8','G':'6','Z':'7','T':'7',
        'Q':'0','D':'0'
    }
    for k,v in repls.items():
        txt = txt.replace(k,v)
    return txt

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙ Grid Control")
    num_rows = st.number_input("Rows", min_value=1, max_value=100, value=25)
    col_mode = st.selectbox("Columns", ["Auto Detect","2","4","6","8"])

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Voucher Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        h, w = processed.shape

        if col_mode != "Auto Detect":
            num_cols_active = int(col_mode)
        else:
            num_cols_active = 4

        col_width = w / num_cols_active
        grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

        results = reader.readtext(processed, detail=1, paragraph=False)

        for (bbox, text, prob) in results:
            if prob < 0.4:
                continue

            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox])

            c_idx = int(cx / col_width)
            r_idx = int((cy / h) * num_rows)

            if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:
                txt = clean_ocr_text(text)
                nums = re.findall(r'\d+', txt)
                if nums:
                    grid_data[r_idx][c_idx] = nums[0]

        st.session_state['data_final'] = grid_data
        st.success("Scan Complete")

# ---------------- DISPLAY ----------------
if 'data_final' in st.session_state:
    st.data_editor(
        st.session_state['data_final'],
        use_container_width=True
    )
