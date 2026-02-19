import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import os
from itertools import permutations

st.set_page_config(page_title="Lottery Pro 2026 Auto Detect", layout="wide")

# ---------------- OCR ----------------
@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- LOAD TEMPLATES ----------------
@st.cache_resource
def load_templates():
    templates = {}
    for i in range(10):
        path = os.path.join("templates", f"{i}.png")
        if os.path.exists(path):
            img = cv2.imread(path, 0)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
            templates[str(i)] = cv2.resize(img, (28, 28))
    return templates

templates = load_templates()

def template_match_digit(roi):
    roi = cv2.resize(roi, (28,28))
    best_score = -1
    best_digit = ""

    for digit, tmpl in templates.items():
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, _ = cv2.minMaxLoc(res)
        if score > best_score:
            best_score = score
            best_digit = digit

    return best_digit if best_score > 0.4 else ""

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
    num_rows = st.number_input("Rows", min_value=1, value=25)
    col_mode = st.selectbox("Columns", ["Auto Detect","2","4","6","8"], index=0)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload Voucher Image", type=["jpg","jpeg","png"])

if uploaded_file:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 OCR Scan"):

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2)
        _, processed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

        h, w = processed.shape

        # ---------- AUTO COLUMN DETECT ----------
        proj = np.sum(processed, axis=0)
        threshold = np.percentile(proj, 75)
        text_mask = proj > threshold
        col_indices = np.where(text_mask)[0]

        clusters = []
        if len(col_indices) > 0:
            current = [col_indices[0]]
            for idx in col_indices[1:]:
                if idx - current[-1] < 15:
                    current.append(idx)
                else:
                    clusters.append(current)
                    current = [idx]
            clusters.append(current)

        num_cols_detected = len(clusters)
        if num_cols_detected < 6:
            num_cols_detected = 6
        if num_cols_detected > 8:
            num_cols_detected = 8

        if col_mode != "Auto Detect":
            num_cols_active = int(col_mode)
        else:
            num_cols_active = num_cols_detected

        col_width = w / num_cols_active
        grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

        # ---------- OCR READ ----------
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

                # NUMBER COLUMN
                if c_idx % 2 == 0:
                    nums = re.findall(r'\d+', txt)

                    if nums:
                        txt = nums[0].zfill(3)
                    else:
                        # fallback to template matching
                        x,y,w1,h1 = cv2.boundingRect(np.array(bbox).astype(int))
                        roi = processed[y:y+h1, x:x+w1]
                        txt = template_match_digit(roi)

                # AMOUNT COLUMN
                else:
                    nums = re.findall(r'\d+', txt)
                    txt = max(nums, key=lambda x: int(x)) if nums else ""

                grid_data[r_idx][c_idx] = txt

        # ---------- DITTO LOGIC ----------
        for c in range(num_cols_active):
            last_val = ""
            for r in range(num_rows):
                curr = grid_data[r][c].strip()

                if curr == "":
                    grid_data[r][c] = last_val
                else:
                    last_val = curr

        st.session_state['data_final'] = grid_data
        st.success(f"Columns Used: {num_cols_active}")

# ---------------- DISPLAY ----------------
if 'data_final' in st.session_state:
    edited_data = st.data_editor(
        st.session_state['data_final'],
        use_container_width=True
    )
