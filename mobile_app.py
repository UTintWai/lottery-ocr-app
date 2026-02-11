import streamlit as st
import numpy as np
import easyocr
import gspread
from PIL import Image
from io import BytesIO
from oauth2client.service_account import ServiceAccountCredentials
import re
import json
from itertools import permutations

# --- Page Setting ---
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# --- Google Credentials ---
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

@st.cache_resource
def load_ocr():
    # 'en' ကို ထည့်ထားမှ R ကို ဖတ်နိုင်မှာပါ
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def expand_r_sorted(text):
    """267R ကို ၆ ကွက်ဖြန့်ပြီး ငယ်စဉ်ကြီးလိုက်စီခြင်း"""
    digits = re.sub(r'\D', '', text)
    if len(digits) == 3:
        perms = set([''.join(p) for p in permutations(digits)])
        return sorted(list(perms))
    return [digits.zfill(3)] if digits else []

st.title("🎰 Lottery OCR (Original Grid System)")

with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက် ရွေးပါ", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"])

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(BytesIO(uploaded_file.read()))
    img_array = np.array(image)
    st.image(image, use_container_width=True)

    if st.button("🔍 AI ဖြင့် ဖတ်မည်"):
        with st.spinner("ဒေတာများကို ခွဲခြားနေပါသည်..."):
            # အရင်အတိုင်း အကုန်ဖတ်နိုင်စေရန် detail=1 နှင့် paragraph=False ပုံစံအတိုင်းသုံးထားသည်
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
                
                # အရင်အတိုင်း Grid Logic
                if col_mode == "၂ တိုင်": c_idx = 0 if x_pos < 0.50 else 1
                elif col_mode == "၄ တိုင်":
                    if x_pos < 0.25: c_idx = 0
                    elif x_pos < 0.50: c_idx = 1
                    elif x_pos < 0.75: c_idx = 2
                    else: c_idx = 3
                elif col_mode == "၆ တိုင်":
                    if x_pos < 0.166: c_idx = 0
                    elif x_pos < 0.333: c_idx = 1
                    elif x_pos < 0.50: c_idx = 2
                    elif x_pos < 0.666: c_idx = 3
                    elif x_pos < 0.833: c_idx = 4
                    else: c_idx = 5
                else: 
                    c_idx = min(7, max(0, int(x_pos * 8)))

                r_idx = int((cy - top_y) // cell_h)
                if 0 <= r_idx < num_rows:
                    # R ပါ ဖတ်နိုင်အောင် clean logic ပြောင်းထားသည်
                    clean = re.sub(r'[^0-9Rr]', '', text.upper())
                    grid_data[r_idx][c_idx] = clean

            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_df = st.data_editor(st.session_state['data_final'], num_rows="dynamic", use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        if creds:
            try:
                client = gspread.authorize(creds)
                ss = client.open("LotteryData")
                
                # Sheet 1: အရင်အတိုင်း Append လုပ်မည်
                sh1 = ss.get_worksheet(0)
                sh1.append_rows(edited_df)
                
                # Sheet 2: ပတ်လည်ဖြန့်ခြင်းနှင့် Sorting စီခြင်း
                sh2 = ss.get_worksheet(1)
                expanded_list = []
                # အတွဲအတွဲလိုက်ယူခြင်း (ဂဏန်းတိုင်၊ ထိုးကြေးတိုင်)
                if col_mode == "၆ တိုင်": pairs = [(0,1), (2,3), (4,5)]
                elif col_mode == "၄ တိုင်": pairs = [(0,1), (2,3)]
                elif col_mode == "၂ တိုင်": pairs = [(0,1)]
                else: pairs = [(0,1), (2,3), (4,5), (6,7)]

                for row in edited_df:
                    for g_col, t_col in pairs:
                        g_val = str(row[g_col])
                        t_val = str(row[t_col])
                        if g_val:
                            if 'R' in g_val:
                                for p in expand_r_sorted(g_val): expanded_list.append([p, t_val])
                            else:
                                expanded_list.append([g_val[-3:].zfill(3) if g_val.isdigit() else g_val, t_val])
                
                # Sheet 2 ထဲက ဂဏန်းတွေကို အငယ်မှအကြီး စီသိမ်းမည်
                expanded_list.sort(key=lambda x: x[0])
                
                if expanded_list:
                    sh2.append_rows(expanded_list)
                st.success("🎉 သိမ်းဆည်းမှု အောင်မြင်ပါသည်။ (Sheet 1 & Sheet 2)")
            except Exception as e:
                st.error(f"Sheet Error: {e}")