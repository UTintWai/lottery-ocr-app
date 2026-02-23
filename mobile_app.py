import streamlit as st
import numpy as np
import cv2
import easyocr
import re

# --- OCR ENGINE ---
@st.cache_resource
def load_ocr():
    # 'en' က ဂဏန်းတွေအတွက် ပိုမြန်ပြီး တိကျပါတယ်
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def pre_process_for_lottery(img):
    # 1. Gray ပြောင်းမယ်
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. ပုံကို ၂ ဆ ချဲ့မယ် (စာလုံးသေးနေရင် OCR ဖတ်ရခက်လို့ပါ)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # 3. Contrast မြှင့်ပြီး Noise ဖယ်မယ်
    dist = cv2.fastNlMeansDenoising(gray, h=10)
    
    # 4. အဖြူအမည်း ပြတ်သားအောင် လုပ်မယ် (Otsu Thresholding)
    _, thresh = cv2.threshold(dist, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def get_lottery_data(img, rows, cols):
    # ပုံကို အရင်ရှင်းအောင်လုပ်မယ်
    processed_img = pre_process_for_lottery(img)
    h, w = processed_img.shape
    
    # OCR ဖတ်မယ် (paragraph=False က တစ်ကွက်ချင်းစီ ဖတ်ဖို့ ပိုကောင်းပါတယ်)
    results = reader.readtext(processed_img, detail=1, paragraph=False)
    
    # Data သိမ်းမယ့် Grid
    grid = [["" for _ in range(cols)] for _ in range(rows)]
    
    for (bbox, text, prob) in results:
        # Bounding Box ရဲ့ ဗဟိုကို ရှာမယ်
        (tl, tr, br, bl) = bbox
        cx = (tl[0] + br[0]) / 2
        cy = (tl[1] + br[1]) / 2
        
        # Grid index တွက်မယ်
        c_idx = int(cx / (w / cols))
        r_idx = int(cy / (h / rows))
        
        if 0 <= r_idx < rows and 0 <= c_idx < cols:
            val = text.strip().upper()
            
            # Ditto Logic (။, ", u, U စတာတွေကို DITTO လို့ ယူမယ်)
            if any(char in val for char in ['"', '။', '=', 'U', 'V', '`', '4']):
                grid[r_idx][c_idx] = "DITTO"
            else:
                # ဂဏန်းသက်သက်ပဲ ယူမယ်
                clean_num = re.sub(r'[^0-9]', '', val)
                if clean_num:
                    grid[r_idx][c_idx] = clean_num.zfill(3)

    # DITTO Fill Down
    for c in range(cols):
        for r in range(1, rows):
            if grid[r][c] == "DITTO":
                grid[r][c] = grid[r-1][c]
                
    return grid

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Lottery Scanner", layout="wide")
st.title("🔢 Lottery Voucher Scanner (6/8 Columns)")

with st.sidebar:
    st.header("Settings")
    col_count = st.selectbox("တိုင်အရေအတွက်", [6, 8], index=1)
    row_count = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    # ပုံဖတ်မယ်
    raw_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    
    st.image(img, caption="မူရင်းပုံ", width=400)
    
    if st.button("🚀 စကင်ဖတ်မယ်"):
        with st.spinner("စာလုံးများကို ဖော်ထုတ်နေပါသည်..."):
            final_data = get_lottery_data(img, row_count, col_count)
            st.session_state['scan_result'] = final_data

if 'scan_result' in st.session_state:
    st.subheader("စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_df = st.data_editor(st.session_state['scan_result'], use_container_width=True)
    
    if st.button("💾 Save to Excel/Sheet"):
        st.success("ဒေတာများကို သိမ်းဆည်းရန် အသင့်ဖြစ်ပါပြီ!")
