import cv2
import easyocr
import numpy as np
import pandas as pd
import re
import os

# ----------------------
# SETTINGS
# ----------------------
image_path = "input.jpg"     # <-- ဒီနေရာမှာ သင့်ပုံနာမည်ထည့်ပါ
num_rows = 25
num_cols = 8                 # 2 / 4 / 6 / 8 ပြောင်းနိုင်ပါတယ်

# ----------------------
# CHECK FILE EXISTS
# ----------------------
if not os.path.exists(image_path):
    print("❌ Image file not found. Check image_path.")
    exit()

# ----------------------
# LOAD IMAGE SAFE
# ----------------------
img = cv2.imread(image_path)

if img is None:
    print("❌ Failed to load image.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------
# INIT OCR
# ----------------------
reader = easyocr.Reader(['en'], gpu=False)

h, w = gray.shape
col_width = w / num_cols
row_height = h / num_rows

grid = []
last_bet_values = [""] * num_cols

# ----------------------
# CELL BY CELL OCR
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

        # Preprocessing
        cell = cv2.resize(cell, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        cell = cv2.GaussianBlur(cell, (3,3), 0)
        cell = cv2.adaptiveThreshold(
            cell, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )

        result = reader.readtext(cell, detail=0)
        text = result[0].strip().upper() if result else ""

        # OCR Common Fix
        text = text.replace("O","0")
        text = text.replace("S","5")
        text = text.replace("I","1")
        text = text.replace("Z","7")
        text = text.replace("X","*")
        text = text.replace("×","*")

        # ----------------------
        # NUMBER COLUMN
        # ----------------------
        if c % 2 == 0:
            digits = re.sub(r'[^0-9]', '', text)
            if len(digits) >= 3:
                row_data[c] = digits[-3:]

        # ----------------------
        # BET COLUMN
        # ----------------------
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
# SAVE TO EXCEL
# ----------------------
columns = [f"Col{i+1}" for i in range(num_cols)]
df = pd.DataFrame(grid, columns=columns)
df.to_excel("output.xlsx", index=False)

print("✅ Finished! Saved as output.xlsx")
