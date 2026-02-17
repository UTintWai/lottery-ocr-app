import streamlit as st
import cv2
import easyocr
import numpy as np
import pandas as pd
import re
from PIL import Image
import io

st.title("Lottery OCR App")

# ----------------------
# SETTINGS
# ----------------------
num_rows = 25
num_cols = st.selectbox("Select Columns", [2,4,6,8], index=3)

# ----------------------
# LOAD OCR MODEL (CACHE)
# ----------------------
@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)

reader = load_reader()

# ----------------------
# FILE UPLOAD
# ----------------------
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    img = np.array(image)

    # Resize large image (RAM save)
    h0, w0 = img.shape[:2]
    max_width = 1200
    if w0 > max_width:
        scale = max_width / w0
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = gray.shape
    col_width = w / num_cols
    row_height = h / num_rows

    grid = []
    last_bet_values = [""] * num_cols

    # ----------------------
    # OCR LOOP
    # ----------------------
    for r in range(num_rows):

        row_data = [""] * num_cols

        for c in range(num_cols):

            x1 = int(c * col_width)
            x2 = int((c + 1) * col_width)
            y1 = int(r * row_height)
            y2 = int((r + 1) * row_height)

            cell = gray[y1:y2, x1:x2]

            if cell.size == 0:
                continue

            cell = cv2.resize(cell, None, fx=1.4, fy=1.4)
            cell = cv2.threshold(cell, 150, 255, cv2.THRESH_BINARY)[1]

            result = reader.readtext(cell, detail=0, paragraph=False)
            text = result[0].strip().upper() if result else ""

            # OCR Fix
            text = text.replace("O","0")
            text = text.replace("S","5")
            text = text.replace("I","1")
            text = text.replace("Z","7")
            text = text.replace("X","*")
            text = text.replace("×","*")

            # NUMBER COLUMN
            if c % 2 == 0:
                digits = re.sub(r'[^0-9]', '', text)
                if len(digits) >= 3:
                    row_data[c] = digits[-3:]

            # BET COLUMN
            else:
                if text == "" or "။" in text:
                    row_data[c] = last_bet_values[c]
                else:
                    clean = re.sub(r'[^0-9*]', '', text)
                    if clean != "":
                        row_data[c] = clean
                        last_bet_values[c] = clean
                    else:
                        row_data[c] = last_bet_values[c]

        grid.append(row_data)

    # ----------------------
    # SHOW RESULT
    # ----------------------
    columns = [f"Col{i+1}" for i in range(num_cols)]
    df = pd.DataFrame(grid, columns=columns)

    st.success("Finished OCR")
    st.dataframe(df)

    # ----------------------
    # CREATE EXCEL IN MEMORY
    # ----------------------
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')

    excel_data = output.getvalue()

    st.download_button(
        label="Download Excel",
        data=excel_data,
        file_name="output.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
