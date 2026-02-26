import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Scanner v16", layout="wide")

@st.cache_resource
def load_ocr():
    # RAM ချွေတာရန်နှင့် တိကျရန် ချိန်ညှိထားသော Reader
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def enhance_image(img):
    # ပုံကို AI ဖတ်ရလွယ်အောင် Contrast နှင့် Brightness ညှိခြင်း
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # အလင်းအမှောင် အလိုအလျောက်ညှိခြင်း (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    return enhanced

def process_v16(img, n_cols):
    h, w = img.shape[:2]
    # RAM crash မဖြစ်စေရန် အရွယ်အစားကို ၁၄၀၀ မှာ ကန့်သတ်ထားသည်
    target_w = 1400
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    processed_gray = enhance_image(img_resized)
    
    # OCR ဖတ်ခြင်း
    results = reader.readtext(processed_gray, paragraph=False, link_threshold=0.2, mag_ratio=1.2)
    
    raw_data = []
    for (bbox, text, prob) in results:
        cx = np.mean([p[0] for p in bbox])
        cy = np.mean([p[1] for p in bbox])
        raw_data.append({'x': cx, 'y': cy, 'text': text.strip().upper()})

    if not raw_data: return []

    # --- ROW CLUSTERING ---
    raw_data.sort(key=lambda k: k['y'])
    rows_list = []
    y_threshold = 28
    
    current_row = [raw_data[0]]
    for i in range(1, len(raw_data)):
        if raw_data[i]['y'] - current_row[-1]['y'] < y_threshold:
            current_row.append(raw_data[i])
        else:
            rows_list.append(current_row)
            current_row = [raw_data[i]]
    rows_list.append(current_row)

    # --- GRID CALCULATION ---
    final_grid = []
    col_width = target_w / n_cols

    for row_items in rows_list:
        row_cells = ["" for _ in range(n_cols)]
        bins = [[] for _ in range(n_cols)]
        
        for item in row_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < n_cols:
                bins[c_idx].append(item)
        
        for c in range(n_cols):
            bins[c].sort(key=lambda k: k['x'])
            combined_txt = "".join([i['text'] for i in bins[c]])
            
            # Ditto Symbols: ။ သို့မဟုတ် အလားတူသင်္ကေတများ
            is_ditto = any(m in combined_txt for m in ['"', '။', '=', '||', 'LL', '`', 'V', '4', 'U', 'Y', '1', '7']) and len(combined_txt) <= 2
            
            if is_ditto:
                row_cells[c] = "DITTO"
            else:
                num = re.sub(r'[^0-9]', '', combined_txt)
                if num:
                    if c % 2 == 0: # ဂဏန်းတိုင်
                        row_cells[c] = num.zfill(3) if len(num) <= 3 else num[:3]
                    else: # ထိုးကြေးတိုင်
                        row_cells[c] = num
        final_grid.append(row_cells)

    # --- AUTO-FILL DITTO & MISSING AMOUNTS ---
    for c in range(n_cols):
        if c % 2 != 0: # ထိုးကြေးတိုင်များတွင်သာ
            last_amt = ""
            for r in range(len(final_grid)):
                val = final_grid[r][c].strip()
                # အကယ်၍ အကွက်လွတ်နေလျှင် သို့မဟုတ် DITTO ဖြစ်နေလျှင် အပေါ်မှကူးမည်
                if val == "DITTO" or val == "":
                    if last_amt != "":
                        final_grid[r][c] = last_amt
                else:
                    last_amt = val
        else: # ဂဏန်းတိုင်များအတွက် Ditto ဖျက်မည်
            for r in range(len(final_grid)):
                if final_grid[r][c] == "DITTO": final_grid[r][c] = ""
                
    return final_grid

# (Google Sheets Function များ အရင်အတိုင်းပဲမို့ ချန်လှပ်ထားပါမယ်)
