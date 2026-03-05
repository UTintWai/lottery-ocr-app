import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# --- CONFIG ---
st.set_page_config(page_title="Lottery Pro v76", layout="wide")
SHEET_NAME = "LotteryData" 

def save_to_gsheet(data_to_save):
    try:
        if not data_to_save: return False
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sheet = client.open(SHEET_NAME).get_worksheet(0)
        formatted_rows = [[f"'{str(c)}" if str(c).strip() != "" else "" for c in row] for row in data_to_save]
        sheet.append_rows(formatted_rows, value_input_option='USER_ENTERED')
        return True
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def process_column(img_col, is_number_col=True):
    """တိုင်တစ်ခုချင်းစီကို OCR ဖတ်ပေးသည့် function"""
    reader = load_ocr()
    # ပုံကို ပိုပြတ်သားအောင် Contrast မြှင့်သည်
    gray = cv2.cvtColor(img_col, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    results = reader.readtext(enhanced, paragraph=False)
    # y position အလိုက် စီသည်
    results.sort(key=lambda x: np.mean([p[1] for p in x[0]]))
    
    data = []
    for (bbox, text, prob) in results:
        y_center = np.mean([p[1] for p in bbox])
        txt = text.upper().replace(' ', '')
        
        # Ditto Check
        if re.search(r'[။၊"=“_…\.\-\']', txt):
            val = "DITTO"
        else:
            txt = txt.replace('S','5').replace('G','6').replace('B','8').replace('I','1').replace('O','0').replace('D','0')
            num = re.sub(r'[^0-9]', '', txt)
            if is_number_col and num:
                val = num.zfill(3)[-3:]
            else:
                val = num
        
        if val:
            data.append({'y': y_center, 'val': val})
    return data

def process_v76(img):
    h, w = img.shape[:2]
    # ပုံကို တိုင် ၄ တိုင် အညီအမျှ ပိုင်းဖြတ်သည်
    col_w = w // 4
    final_rows = []
    
    # တိုင်တစ်ခုချင်းစီကို Scan ဖတ်သည်
    all_cols_data = []
    for i in range(4):
        crop = img[:, i*col_w : (i+1)*col_w]
        is_num = (i % 2 == 0)
        col_data = process_column(crop, is_number_col=is_num)
        all_cols_data.append(col_data)

    # 🎯 ROW ALIGNMENT: Y-coordinate ကို သုံးပြီး တိုင် ၄ တိုင်က ဒေတာတွေကို ပြန်ပေါင်းသည်
    # စုစုပေါင်း ၂၅ တန်းဝန်းကျင် ရစေရန် Y-bins (0 to h) ခွဲခြားသည်
    bins = np.linspace(0, h, 27) # ၂၅ တန်း + Margin
    structured_data = [["" for _ in range(4)] for _ in range(len(bins)-1)]
    
    for col_idx, col_items in enumerate(all_cols_data):
        for item in col_items:
            # ဘယ် bin ထဲမှာ ရှိသလဲ ရှာသည်
            bin_idx = np.digitize(item['y'], bins) - 1
            if 0 <= bin_idx < len(structured_data):
                structured_data[bin_idx][col_idx] = item['val']

    # အလွတ်အတန်းများကို ဖယ်ထုတ်သည်
    cleaned_data = [r for r in structured_data if any(c != "" for c in r)]
    
    # Ditto Filling (ထိုးကြေးတိုင်များအတွက်)
    for c in [1, 3]:
        last = ""
        for r in range(len(cleaned_data)):
            if cleaned_data[r][c].isdigit(): last = cleaned_data[r][c]
            elif (cleaned_data[r][c] == "DITTO" or cleaned_data[r][c] == "") and last != "":
                cleaned_data[r][c] = last
                
    return cleaned_data

# --- UI ---
st.title("🔢 Lottery Pro v76 (Column-wise Scanning)")
st.info("တိုင်တစ်ခုချင်းစီကို အသေးစိတ်ဖြတ်ထုတ်ဖတ်သည့်စနစ်ဖြစ်ပါသည်။")

c1, c2 = st.columns(2)
with c1: up_l = st.file_uploader("ဘယ်ဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="l")
with c2: up_r = st.file_uploader("ညာဘက် ၄ တိုင်", type=['jpg', 'png', 'jpeg'], key="r")

if st.button("🔍 Precise Scan"):
    with st.spinner('တစ်တိုင်ချင်းစီကို အသေးစိတ် စစ်ဆေးနေပါသည်...'):
        l_res, r_res = [], []
        if up_l: l_res = process_v76(cv2.imdecode(np.frombuffer(up_l.read(), np.uint8), 1))
        if up_r: r_res = process_v76(cv2.imdecode(np.frombuffer(up_r.read(), np.uint8), 1))
            
        max_r = max(len(l_res), len(r_res))
        final = []
        for i in range(max_r):
            l = l_res[i] if i < len(l_res) else ["","","",""]
            r = r_res[i] if i < len(r_res) else ["","","",""]
            final.append(l + r)
        
        st.session_state['data_v76'] = final
        st.rerun()

if 'data_v76' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား")
    st.session_state['data_v76'] = st.data_editor(st.session_state['data_v76'], use_container_width=True, num_rows="dynamic")
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_gsheet(st.session_state['data_v76']):
            st.success("✅ သိမ်းဆည်းပြီးပါပြီ!")
