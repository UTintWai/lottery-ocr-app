import streamlit as st
import cv2
import numpy as np
import easyocr
import gspread
from PIL import Image
from oauth2client.service_account import ServiceAccountCredentials

# Mobile Screen Layout
st.set_page_config(page_title="Lottery Mobile OCR", layout="centered")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.title("🎰 Lottery OCR Mobile")
st.info("ဖုန်းဖြင့် ပုံရိုက်ပြီး ၂၊ ၄၊ ၆၊ ၈ တိုင်စနစ်ဖြင့် ဖတ်နိုင်ပါသည်")

# Settings
st.sidebar.header("Grid Settings")
num_cols = st.sidebar.selectbox("တိုင်အရေအတွက် ရွေးပါ", [2, 4, 6, 8], index=2)
num_rows = st.sidebar.number_input("အတန်းအရေအတွက်", value=25)

# Camera/File Upload
uploaded_file = st.file_uploader("ပုံရိုက်ရန် သို့မဟုတ် ရွေးရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, caption="တင်ထားသောပုံ", use_container_width=True)

    if st.button("🚀 AI ဖြင့် ဖတ်မည်", use_container_width=True):
        with st.spinner("AI ဖတ်နေပါသည်..."):
            # Image Processing for accuracy
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            results = reader.readtext(gray)
            
            h, w = gray.shape[:2]
            grid_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            if results:
                y_pts = [res[0][0][1] for res in results]
                top_y = max(0, min(y_pts) - 10)
                cell_w, cell_h = w / num_cols, (h - top_y) / num_rows

                for (bbox, text, prob) in results:
                    cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                    c_idx = int(cx // cell_w)
                    r_idx = int((cy - top_y) // cell_h)
                    
                    if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                        clean = "".join(filter(str.isdigit, text))
                        if clean:
                            if num_cols >= 4 and c_idx % 2 == 0: clean = clean.zfill(3)[-3:]
                            grid_data[r_idx][c_idx] = clean
                        elif any(m in text.lower() for m in ['။', '"', 'u', 'v']):
                            grid_data[r_idx][c_idx] = "DITTO"

                # Ditto logic
                for c in range(num_cols):
                    last_val = ""
                    for r in range(num_rows):
                        if grid_data[r][c].isdigit(): last_val = grid_data[r][c]
                        elif grid_data[r][c] == "DITTO" and last_val != "":
                            grid_data[r][c] = last_val
            
            st.session_state['result'] = grid_data

# Edit and Upload
if 'result' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    # အကွက်ထဲမှာတင် လက်နဲ့နှိပ်ပြီး ပြင်နိုင်သည်
    final_data = st.data_editor(st.session_state['result'], num_rows="dynamic")

    if st.button("✅ Google Sheet သို့ ပို့မည်", use_container_width=True):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
            client = gspread.authorize(creds)
            sheet = client.open("LotteryData").sheet1
            sheet.clear()
            sheet.update("A1", final_data)
            st.success("Google Sheet ထဲသို့ ဒေတာများ အောင်မြင်စွာ ပို့ပြီးပါပြီ။")
            st.balloons()
        except Exception as e:
            st.error(f"Error: {str(e)}")