import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- ၁။ CONFIG & FUNCTIONS ----------------
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

@st.cache_resource
def load_ocr():
    # GPU မရှိလျှင် False ထားပါ၊ စာလုံးအစိပ်ဆုံးဖတ်ရန် English တစ်ခုတည်းသုံးပါ
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3: return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

def process_bet_logic(num_txt, amt_txt):
    clean_num = re.sub(r'[^0-9R]', '', str(num_txt).upper())
    amt_str = str(amt_txt).upper().replace('X','*')
    results = {}
    try:
        if 'R' in clean_num:
            base = clean_num.replace('R','')
            perms = get_all_permutations(base)
            num_part = re.sub(r'\D','',amt_str)
            amt = int(num_part) if num_part else 0
            if perms and amt > 0:
                split = amt // len(perms)
                for p in perms: results[p] = split
        elif '*' in amt_str:
            parts = amt_str.split('*')
            if len(parts)==2:
                base_amt = int(re.sub(r'\D','',parts[0]))
                total_amt = int(re.sub(r'\D','',parts[1]))
                num_final = clean_num.zfill(3)
                results[num_final] = base_amt
                perms = [p for p in get_all_permutations(num_final) if p!=num_final]
                if perms:
                    split = (total_amt-base_amt)//len(perms)
                    for p in perms: results[p] = split
        else:
            num_part = re.sub(r'\D','',amt_str)
            amt = int(num_part) if num_part else 0
            num_final = clean_num.zfill(3) if (clean_num.isdigit() and len(clean_num)<=3) else clean_num
            if num_final: results[num_final] = amt
    except: pass
    return results

# ---------------- ၂။ SIDEBAR ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    bet_limit = st.number_input("Limit (ပိုလျှံတန်ဖိုးသတ်မှတ်ရန်)", min_value=100, value=5000)
    num_rows = st.number_input("Rows (စာကြောင်းအရေအတွက်)", min_value=1, value=25)
    col_mode = st.selectbox("Columns (တိုင်အရေအတွက်)", ["2","4","6","8"], index=3) # Default 8
    num_cols_active = int(col_mode)

# ---------------- ၃။ OCR SCAN LOGIC ----------------
st.title("🎰 Lottery OCR 8-Column Stable")
uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    # ---------------- ၃။ OCR SCAN LOGIC (ကျဲခြင်းကို ကာကွယ်သော Dynamic Splitting Version) ----------------
if st.button("🔍 စစ်ဆေးမည် (OCR Scan)"):
    with st.spinner(f"{num_cols_active} တိုင်စလုံးကို အကွက်စိပ်စိပ် ပြန်စီနေပါသည်..."):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # စာလုံးပိုကြွလာအောင် Contrast မြှင့်တင်ခြင်း
            processed_img = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
            
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(num_cols_active)] for _ in range(num_rows)]
            
            results = reader.readtext(processed_img, detail=1)

            # Column တစ်ခုချင်းစီရဲ့ boundary တွေကို တိတိကျကျ သတ်မှတ်ခြင်း
            col_bounds = [i * (w / num_cols_active) for i in range(num_cols_active + 1)]

            for (bbox, text, prob) in results:
                if prob < 0.2: continue
                
                # စာလုံးစု၏ ဘယ်ဘက်၊ ညာဘက် နှင့် အမြင့်ကို ယူခြင်း
                x_min = bbox[0][0]
                x_max = bbox[1][0]
                y_center = np.mean([p[1] for p in bbox])
                
                r_idx = int(y_center / (h / num_rows))
                
                # စာသားအစုအဝေးက Column ဘယ်နှစ်ခုစာ ကျော်နေသလဲ စစ်ဆေးခြင်း
                if 0 <= r_idx < num_rows:
                    # စာလုံးက တစ်တိုင်ထက်ပိုကျော်နေရင် ခွဲထုတ်မည်
                    words = text.split() if " " in text else [text]
                    
                    for i, part in enumerate(words):
                        # စာလုံးတစ်လုံးချင်းစီရဲ့ ခန့်မှန်းခြေ x-position
                        estimated_cx = x_min + (i * (x_max - x_min) / len(words))
                        
                        # ဘယ် Column ထဲ ကျသလဲ ရှာခြင်း
                        c_idx = -1
                        for b in range(num_cols_active):
                            if col_bounds[b] <= estimated_cx < col_bounds[b+1]:
                                c_idx = b
                                break
                        
                        if c_idx != -1:
                            txt = part.upper().strip()
                            # Character Fixes
                            repls = {'O':'0','I':'1','S':'5','G':'6','Z':'7','B':'8','A':'4','T':'7','L':'1'}
                            for k, v in repls.items(): txt = txt.replace(k, v)
                            
                            # Clean based on Column Type
                            if c_idx % 2 == 0: 
                                txt = re.sub(r'[^0-9R]', '', txt)
                            else: 
                                txt = re.sub(r'[^0-9X*]', '', txt)
                            
                            # ဒေတာထည့်ခြင်း (ရှိပြီးသားဖြစ်ပါက ကော်မာခြား၍ ပေါင်းမည်)
                            if grid_data[r_idx][c_idx] == "":
                                grid_data[r_idx][c_idx] = txt
                            else:
                                # နံပါတ်တိုင်ဆိုလျှင် မပေါင်းဘဲ အသစ်ပဲယူမည်၊ ပမာဏဆိုလျှင် ပေါင်းမည်
                                grid_data[r_idx][c_idx] = txt if c_idx % 2 == 0 else grid_data[r_idx][c_idx] + txt

            # Ditto (") Logic အပေါ်ကတန်ဖိုးယူခြင်း
            for c in range(num_cols_active):
                last_v = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    if curr in ['"', "''", "v", "V", "11", "ll", "LL", "-", "Y", "4"] and last_v:
                        grid_data[r][c] = last_v
                    elif curr: last_v = curr

            st.session_state['data_final'] = grid_data
            st.rerun()
            
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")