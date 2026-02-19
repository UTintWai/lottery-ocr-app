import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import os

st.set_page_config(page_title="Lottery Pro 2026 Stable OCR", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- LOAD DIGIT TEMPLATES ----------------
@st.cache_resource
def load_templates():
    templates = {}
    template_dir = "templates"
    if not os.path.exists(template_dir):
        os.makedirs(template_dir)
    for i in range(10):
        path = os.path.join(template_dir, f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, 0)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            templates[str(i)] = cv2.resize(img, (28,28))
    return templates

templates = load_templates()

def template_match_digit(roi):
    if roi.size == 0:
        return ""
    roi = cv2.resize(roi, (28,28))
    best_score = -1
    best_digit = ""
    for digit, tmpl in templates.items():
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best_digit = digit
    return best_digit if best_score > 0.35 else ""   # cutoff tune

# ---------------- CLEAN OCR ----------------
def clean_ocr_text(txt):
    txt = txt.upper().strip()
    repls = {'O':'0','I':'1','L':'1','S':'5','B':'8','G':'6','Z':'7','T':'7','Q':'0','D':'0'}
    for k,v in repls.items():
        txt = txt.replace(k,v)
    return txt

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("### ⚙ Grid Control")
    num_rows = st.number_input("Rows", min_value=1, max_value=100, value=25)
    col_mode = st.selectbox("Columns", ["Auto Detect","2","4","6","8"], index=0)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Voucher Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):

        # -------- Image Processing --------
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        processed = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        st.image(processed, caption="Processed Image")

        # -------- Grid Setup --------
        h, w = processed.shape
        num_cols_active = int(col_mode) if col_mode != "Auto Detect" else 8
        col_width = w / num_cols_active
        grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

        # -------- OCR & Template Matching --------
        results = reader.readtext(processed, detail=1, paragraph=False)

        for (bbox, text, prob) in results:
            if prob < 0.35:   # cutoff tune
                x, y, w_box, h_box = cv2.boundingRect(np.array(bbox).astype(int))
                roi = processed[y:y+h_box, x:x+w_box]
                text = template_match_digit(roi)
            else:
                text = clean_ocr_text(text)

            cx = np.mean([p[0] for p in bbox])
            cy = np.mean([p[1] for p in bbox])
            c_idx = int(cx / col_width)
            r_idx = int((cy / h) * num_rows)

            if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:
                nums = re.findall(r'\d+', text)
                if nums:
                    grid_data[r_idx][c_idx] = nums[0]

        # -------- Ditto Logic --------
        for c in range(num_cols_active):
            last_val = ""
            for r in range(num_rows):
                if grid_data[r][c] == "":
                    grid_data[r][c] = last_val
                else:
                    last_val = grid_data[r][c]

        st.session_state['data_final'] = grid_data
        st.success(f"OCR Scan Complete using {num_cols_active} columns")

# ---------------- DISPLAY GRID ----------------
if 'data_final' in st.session_state:
    st.data_editor(st.session_state['data_final'], use_container_width=True)
