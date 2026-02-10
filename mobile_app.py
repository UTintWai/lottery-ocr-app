import streamlit as st
import numpy as np
import easyocr
import gspread
from PIL import Image
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
import os
import re
import json

# --- Page Setting ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- Google Credentials ---
creds = None
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Secret ဖတ်သည့်အပိုင်းကို Error မတက်အောင် သေချာပြင်ထားပါသည်
if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
    try:
        secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
        if "private_key" in secret_info:
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
    except Exception as e:
        st.error(f"Secret Error: {e}")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
st.title("🎰 Lottery OCR (2, 4, 8 Columns)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက် ရွေးပါ", ["၂ တိုင်", "၄ တိုင်", "၈ တိုင်"])

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.read()))
    img_array = np.array(image)
    st.image(image, use_container_width=True)

    if st.button("🔍 AI ဖြင့် ဖတ်မည်"):
        with st.spinner("ဒေတာများကို ခွဲခြားနေပါသည်..."):
            results = reader.readtext(img_array)
            h, w = img_array.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            y_pts = sorted([res[0][0][1] for res in results])
            top_y = y_pts[0] if y_pts else 0
            bot_y = y_pts[-1] if y_pts else h
            cell_h = (bot_y - top_y) / (num_rows if num_rows > 0 else 1)

            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                x_pos = cx / w
                
                # --- တိုင်အလိုက် တည်နေရာ Logic (အမှန်ဆုံးပုံစံ) ---
                if col_mode == "၂ တိုင်":
                    c_idx = 0 if x_pos < 0.50 else 1
                elif col_mode == "၄ တိုင်":
                    if x_pos < 0.25: c_idx = 0
                    elif x_pos < 0.50: c_idx = 1
                    elif x_pos < 0.75: c_idx = 2
                    else: c_idx = 3
                else: # ၈ တိုင်
                    c_idx = int(x_pos * 8)
                    c_idx = min(7, max(0, c_idx))

                r_idx = int((cy - top_y) // cell_h)
                if 0 <= r_idx < num_rows:
                    clean = text.strip()
                    has_digit = any(char.isdigit() for char in clean)
                    grid_data[r_idx][c_idx] = "DITTO" if not has_digit and len(clean) > 0 else clean

            # --- Auto-fill & Format Logic ---
            last_valid = [""] * 8
            for r in range(num_rows):
                for c in range(8):
                    val = str(grid_data[r][c])
                    if val in ["DITTO", ""]:
                        grid_data[r][c] = last_valid[c]
                    else:
                        digits = re.sub(r'\D', '', val)
                        # ဂဏန်းတိုင်များ (A, C, E, G) ကို ၃ လုံးညှိမည်
                        if c in [0, 2, 4, 6] and digits:
                            grid_data[r][c] = digits[-3:].zfill(3)
                        else:
                            grid_data[r][c] = digits
                        last_valid[c] = grid_data[r][c]

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_df = st.data_editor(st.session_state['data_final'], num_rows="dynamic", use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                sheet = client.open("LotteryData").sheet1
                sheet.clear()
                sheet.update("A1", edited_df, value_input_option="RAW")
                st.success("🎉 Google Sheet သို့ ပို့ပြီးပါပြီ။")
            except Exception as e:
                st.error(f"Sheet Error: {e}")