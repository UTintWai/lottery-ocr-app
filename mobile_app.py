import streamlit as st
import numpy as np
import easyocr
import gspread
from PIL import Image
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
import json

st.set_page_config(
    page_title="Lottery Pro 2026",
    page_icon="lll.png",
    layout="wide"
)

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
st.title("lll.png Lottery OCR Pro")

# JSON key file ကို ဖုန်း storage ထဲမှာ ထည့်ထားပြီး path ကို သုံးပါ
with open("/storage/emulated/0/service_account.json") as f:   # ဖုန်းထဲမှာ JSON key file path
    creds_info = json.load(f)

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    num_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_data = uploaded_file.read()
    image = Image.open(BytesIO(img_data))
    img_array = np.array(image)
    st.image(image, use_container_width=True)

    if st.button("🔍 AI ဖြင့် ဖတ်မည်"):
        with st.spinner("AI ဖတ်နေပါသည်..."):
            results = reader.readtext(img_array)
            h, w = img_array.shape[:2]
            grid_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            y_pts = [res[0][0][1] for res in results] if results else [0]
            top_y = max(0, min(y_pts) - 10)
            cell_w, cell_h = w / num_cols, (h - top_y) / num_rows

            last_values = [""] * num_cols
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx, r_idx = int(cx // cell_w), int((cy - top_y) // cell_h)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    clean = text.strip().replace(" ", "")
                    ditto_list = ["။", "||", "11", "II", "=", "—", "..", "::", "1/"]
                    if any(sym in clean for sym in ditto_list):
                        grid_data[r_idx][c_idx] = "DITTO_MARK"
                    else:
                        grid_data[r_idx][c_idx] = clean

            # Auto-fill Logic
            for r in range(num_rows):
                for c in range(num_cols):
                    if grid_data[r][c] == "DITTO_MARK":
                        grid_data[r][c] = last_values[c]
                    elif grid_data[r][c] != "":
                        last_values[c] = grid_data[r][c]
            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_data = st.data_editor(st.session_state['data_final'], num_rows="dynamic")

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
            client = gspread.authorize(creds)
            sheet = client.open("LotteryData").sheet1
            sheet.clear()
            sheet.update("A1", edited_data)
            st.success("အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ။")
            st.balloons()
        except Exception as e:
            st.error(f"Error: {str(e)}")
