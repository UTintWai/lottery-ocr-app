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

# --- Google Credentials Setup ---
creds = None
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

if "GCP_SERVICE_ACCOUNT_FILE" in st.secrets:
    try:
        secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
        if "private_key" in secret_info:
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
        creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
    except Exception as e:
        st.error(f"Secret Error: {e}")
elif os.path.exists("credentials.json"):
    try:
        creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
    except Exception as e:
        st.error(f"JSON File Error: {e}")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
st.title("🎰 Lottery OCR (4 Columns Format)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.read()))
    img_array = np.array(image)
    st.image(image, use_container_width=True)

    if st.button("🔍 AI ဖြင့် တိကျစွာဖတ်မည်"):
        with st.spinner("တိုင်များအလိုက် ဒေတာများကို ခွဲခြားနေပါသည်..."):
            results = reader.readtext(img_array)
            h, w = img_array.shape[:2]
            # ဂဏန်း ၄ တိုင်အတွက် Column 0, 2, 4, 6 ကို သုံးပါမယ်
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            
            y_pts = sorted([res[0][0][1] for res in results])
            top_y = y_pts[0] if y_pts else 0
            bot_y = y_pts[-1] if y_pts else h
            cell_h = (bot_y - top_y) / (num_rows - 0.5)

            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                x_pos = cx / w
                
                # --- အတိုင် ၄ တိုင် (ဂဏန်းတိုင်များ) ခွဲခြားမှု ---
                # --- အတိုင် ၄ တိုင် (ဂဏန်းတိုင်များ) တည်နေရာကို ပုံနှင့်ကွက်တိဖြစ်အောင် ညှိခြင်း ---
                if x_pos < 0.20: c_idx = 0        # ပထမတိုင် (Column 2)
                elif x_pos < 0.45: c_idx = 2      # ဒုတိယတိုင် (Column 4)
                elif x_pos < 0.70: c_idx = 4      # တတိယတိုင် (Column 6)
                else: c_idx = 6                   # စတုတ္ထတိုင် (Column 8)

                r_idx = int((cy - top_y) // cell_h)
                if 0 <= r_idx < num_rows:
                    clean = text.strip().replace(" ", "")
                    has_digit = any(char.isdigit() for char in clean)
                    grid_data[r_idx][c_idx] = "DITTO_MARK" if not has_digit and len(clean) > 0 else clean

            # --- Auto-fill & 3-Digit Logic (၄ တိုင်တည်းအတွက်) ---
            last_valid = [""] * 8
            for r in range(num_rows):
                for c in [0, 2, 4, 6]: # ဂဏန်းတိုင် ၄ တိုင်ကိုပဲ စစ်ဆေးမယ်
                    if grid_data[r][c] in ["DITTO_MARK", ""]:
                        grid_data[r][c] = last_valid[c]
                    else:
                        # ဂဏန်းမဟုတ်တာတွေဖယ်ပြီး ၃ လုံးဖြစ်အောင် ဖြည့်တယ် (ဥပမာ 5 -> 005)
                        digits = re.sub(r'\D', '', str(grid_data[r][c]))
                        if digits: 
                            grid_data[r][c] = digits.zfill(3)
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
                st.success("🎉 ဒေတာများအားလုံး Google Sheet သို့ ပို့ပြီးပါပြီ။")
                st.balloons()
            except Exception as e:
                st.error(f"⚠️ Google Sheet Error: {str(e)}")
        else:
            st.error("❌ Credentials မရှိပါ။ Secret သို့မဟုတ် JSON ဖိုင်ကို စစ်ဆေးပါ။")