import streamlit as st
import cv2
import numpy as np
import easyocr
import pandas as pd
import re

st.set_page_config(layout="wide")
st.title("📄 Lottery Voucher OCR Scanner")

reader = easyocr.Reader(['en'], gpu=False)

# ---------- CLEAN OCR TEXT ----------
def clean_ocr_text(text):
    text = text.replace("I", "1")
    text = text.replace("l", "1")
    text = text.replace("|", "1")
    text = text.replace("O", "0")
    text = text.replace("o", "0")
    text = text.replace("S", "5")
    text = text.replace("s", "5")
    return text.strip()


# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("Upload Voucher Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 OCR Scan"):

        with st.spinner("Processing..."):

            # ---------- PREPROCESS ----------
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            processed = clahe.apply(gray)

            h, w = processed.shape

            # ---------- AUTO COLUMN DETECT ----------
            proj = np.sum(processed < 200, axis=0)  # dark pixels projection
            threshold = np.mean(proj) * 1.2
            peaks = np.where(proj > threshold)[0]

            clusters = []
            current_cluster = [peaks[0]] if len(peaks) > 0 else []

            for p in peaks[1:]:
                if p - current_cluster[-1] < 10:
                    current_cluster.append(p)
                else:
                    clusters.append(current_cluster)
                    current_cluster = [p]

            if current_cluster:
                clusters.append(current_cluster)

            num_cols_active = len(clusters)
            if num_cols_active == 0:
                num_cols_active = 6  # fallback

            col_width = w / num_cols_active
            num_rows = 20  # adjust if needed

            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]

            # ---------- OCR ----------
            results = reader.readtext(processed, detail=1, paragraph=False)

            for (bbox, text, prob) in results:

                if prob < 0.40:
                    continue

                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])

                c_idx = int(cx / col_width)
                r_idx = int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols_active:

                    txt = clean_ocr_text(text)

                    if c_idx % 2 == 0:  # number column
                        nums = re.findall(r'\d+', txt)
                        txt = nums[0].zfill(3) if nums else ""
                    else:  # amount column
                        nums = re.findall(r'\d+', txt)
                        txt = max(nums, key=lambda x: int(x)) if nums else ""

                    grid_data[r_idx][c_idx] = txt

            # ---------- DITTO LOGIC ----------
            for c in range(num_cols_active):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()

                    if c % 2 == 0:
                        curr = re.sub(r'[^0-9]', '', curr)
                        if curr.isdigit():
                            curr = curr.zfill(3)
                    else:
                        nums = re.findall(r'\d+', curr)
                        curr = max(nums, key=lambda x: int(x)) if nums else ""

                    grid_data[r][c] = curr if curr else last_val

                    if curr:
                        last_val = curr

            # ---------- DISPLAY ----------
            df = pd.DataFrame(grid_data)
            st.success(f"Detected Columns: {num_cols_active}")
            st.dataframe(df, use_container_width=True)
