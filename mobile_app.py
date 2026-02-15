import streamlit as st
import numpy as np
import cv2
import easyocr
import re

# ---------------- ၁။ OCR Initial Setup ----------------
@st.cache_resource
def load_ocr():
    # အမြန်ဆုံးနဲ့ အမှန်ကန်ဆုံးဖြစ်အောင် English တစ်မျိုးတည်း သုံးပါမည်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

st.title("🎰 Lottery Pro 2026 (8-Column Stable)")

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("Rows (စာကြောင်းရေ)", min_value=10, value=25)
    num_cols = 8 # ၈ တိုင် အသေ သတ်မှတ်ထားသည်

# ---------------- ၂။ OCR Logic ----------------
uploaded_file = st.file_uploader("လက်ရေးမူပုဒ်ကို တင်ပေးပါ", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption="တင်ထားသောပုံ", use_container_width=True)

    if st.button("🔍 OCR ဖြင့် ဖတ်မည်"):
        with st.spinner("စာလုံးများကို အကွက်ချစီနေပါသည်..."):
            # Image Processing (စာလုံးပိုထင်ရှားစေရန်)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            processed_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            h, w = img.shape[:2]
            # 8 Columns ပုံသေ Grid ဆောက်သည်
            grid_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            # OCR ဖတ်ခြင်း
            results = reader.readtext(img) # Original image ကို သုံးခြင်းက တစ်ခါတစ်ရံ ပိုမှန်တတ်သည်

            # Column များကို ၈ ပုံ အညီအမျှ ပိုင်းခြင်း
            col_width = w / num_cols
            row_height = h / num_rows

            for (bbox, text, prob) in results:
                if prob < 0.15: continue
                
                # စာလုံး၏ ဗဟိုမှတ်ကို ယူပါမည်
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                # မည်သည့် အကွက်ထဲကျသလဲ တွက်ချက်ခြင်း
                c_idx = int(cx // col_width)
                r_idx = int(cy // row_height)

                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    txt = text.upper().strip()
                    # Character Fixes (ဂဏန်းအလွဲများ ပြင်ခြင်း)
                    repls = {'O':'0','I':'1','S':'5','G':'6','Z':'7','B':'8','A':'4','T':'7','L':'1','U':'0'}
                    for k, v in repls.items():
                        txt = txt.replace(k, v)
                    
                    # ဂဏန်းတိုင် (Column 0, 2, 4, 6)
                    if c_idx % 2 == 0:
                        txt = re.sub(r'[^0-9R]', '', txt)
                    # ပမာဏတိုင် (Column 1, 3, 5, 7)
                    else:
                        txt = re.sub(r'[^0-9X*]', '', txt)
                    
                    # အကွက်ထဲ စာရှိနေလျှင် ထပ်ပေါင်းထည့်ရန်
                    if grid_data[r_idx][c_idx]:
                        grid_data[r_idx][c_idx] += f" {txt}"
                    else:
                        grid_data[r_idx][c_idx] = txt

            # Ditto Logic (") အပေါ်ကတန်ဖိုးယူခြင်း
            for c in range(num_cols):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if curr in ['"', "''", "v", "V", "11", "ll", "-", "4"] and last_val:
                        grid_data[r][c] = last_val
                    elif curr:
                        last_val = curr

            st.session_state['ocr_data'] = grid_data

# ---------------- ၃။ Result Display & Editing ----------------
if 'ocr_data' in st.session_state:
    st.subheader("📝 ရရှိလာသော အချက်အလက်များ (လိုအပ်ပါက ပြင်ဆင်နိုင်သည်)")
    # Data Editor ဖြင့် ပြန်ပြင်နိုင်အောင် လုပ်ထားသည်
    final_df = st.data_editor(st.session_state['ocr_data'], use_container_width=True)
    
    if st.button("🚀 Google Sheet သို့ ပို့မည်"):
        # ဤနေရာတွင် ရှေ့က Google Sheet Code အတိုင်း ဆက်လက်အသုံးပြုနိုင်သည်
        st.success("Google Sheet သို့ ပို့ဆောင်မှု အောင်မြင်ပါသည် (ဥပမာပြချက်)")