import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# ---------------- ၁။ PAGE CONFIGURATION ----------------
st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

# ---------------- ၂။ OCR MODEL LOADING ----------------
@st.cache_resource
def load_ocr():
    # GPU မရှိပါက False ထားပါ
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()

# ---------------- ၃။ PERMUTATION LOGIC (ပတ်လည်တွက်ရန်) ----------------
def get_all_permutations(num_str):
    num_only = re.sub(r'\D', '', num_str)
    if len(num_only) != 3:
        return [num_only] if num_only else []
    return sorted(list(set([''.join(p) for p in permutations(num_only)])))

# ---------------- ၄။ BETTING LOGIC (R စနစ်နှင့် မြှောက်လဒ်စနစ်) ----------------
def process_ocr_results(results, h, w, num_rows, num_cols_active):
    grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
    
    # ၁။ OCR ကရတဲ့ စာသားတွေကို သက်ဆိုင်ရာ အကွက်ထဲ အရင်ထည့်မယ်
    for (bbox, text, prob) in results:
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                rel_x = cx / w
                c_idx = 0
                for i, step in enumerate(col_steps):
                    if rel_x <= step:
                        c_idx = i
                        break
                
                r_idx = int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    txt = text.upper().strip()

                    # ၁။ အင်္ဂလိပ်စာလုံးများကို ဂဏန်းသို့ အတင်းအကျပ်ပြောင်းလဲခြင်း (Hard Mapping)
                    # ဥပမာ- GO -> 60, TZO -> 770 ဖြစ်သွားအောင် လုပ်ပေးပါသည်
                    repls = {
                        'S': '5', 'T': '7', 'Z': '7', 'G': '6', 'I': '1', 
                        'L': '1', 'O': '0', 'B': '8', 'Q': '0', 'A': '4'
                    }
                    for k, v in repls.items():
                        txt = txt.replace(k, v)

                    # ၂။ သုံးလုံးထိုးဂဏန်းတိုင်များ (A, C, E, G) အတွက် အထူးသန့်စင်ခြင်း
                    if c_idx in [0, 2, 4, 6]:
                        # ဂဏန်း (0-9) နှင့် 'R' မှလွဲ၍ ကျန်သည့် စာလုံးအားလုံး (ဥပမာ- [, _, /) ကို ဖယ်ထုတ်ပစ်မည်
                        txt = re.sub(r'[^0-9R]', '', txt)
                        
                        # ဂဏန်း ၃ လုံးထက် ကျော်နေပါက ရှေ့ဆုံး ၃ လုံးကိုသာ ယူမည် (လက်ရေးကပ်နေလျှင်)
                        if len(txt) > 3 and 'R' not in txt:
                            txt = txt[:3]

                    # ၃။ ငွေပမာဏတိုင်များ (B, D, F, H) အတွက် သန့်စင်ခြင်း
                    else:
                        # ဂဏန်း၊ X နှင့် * မှလွဲ၍ ကျန်တာဖယ်မည် (ဥပမာ- [20 ကို 20 ဟု ပြင်မည်)
                        txt = re.sub(r'[^0-9X*]', '', txt)

                    grid_data[r_idx][c_idx] = txt

    # ၂။ Ditto logic (အောက်က အတူတူပဲဆိုတဲ့ သင်္ကေတ) ကို ကိုင်တွယ်ခြင်း
    for c in range(num_cols_active):
        last_valid_val = ""
        for r in range(num_rows):
            curr = grid_data[r][c].strip()
            
            # အကယ်၍ အကွက်က လွတ်နေရင် သို့မဟုတ် " (ditto) သင်္ကေတနဲ့ တူတာတွေ့ရင်
            # လက်ရေးမှာ "4" လိုမျိုး ရေးတတ်တဲ့အတွက် အက္ခရာ/ဂဏန်း မဟုတ်တာတွေကို စစ်ဆေး
            is_ditto = curr in ['"', '""', "''", "4", "ll", "y"] or (not curr.isalnum() and curr != "")
            
            if (curr == "" or is_ditto) and last_valid_val != "":
                grid_data[r][c] = last_valid_val
            elif curr != "":
                # ဂဏန်းအတိုင်ဖြစ်ရင် ၃ လုံးပဲ ယူမယ်
                if c % 2 == 0: 
                    nums_only = re.sub(r'[^0-9R]', '', curr)
                    grid_data[r][c] = nums_only
                else:
                    grid_data[r][c] = curr
                last_valid_val = grid_data[r][c]
                
    return grid_data

# ---------------- ၅။ SIDEBAR SETTINGS ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက် (Rows)", min_value=1, value=25)
    col_mode = st.selectbox("အတိုင်အရေအတွက် (Columns)", ["2", "4", "6", "8"], index=3)
    num_cols_active = int(col_mode)
    st.divider()
    st.info("Logic: 267R-360 ဆိုလျှင် ၆ ကွက်ကို ၆၀ စီ ခွဲပေးပါမည်။")

# ---------------- ၆။ MAIN UI & OCR SCAN ----------------
st.title("🎰 Lottery OCR Stable Version 2026")

uploaded_file = st.file_uploader("📥 လက်ရေးမူပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, channels="BGR", use_container_width=True)

    if st.button("🔍 စစ်ဆေးမည် (OCR Scan)"):
        with st.spinner("၈ တိုင်စလုံးကို အမြင့်ဆုံး တိကျမှုဖြင့် ဖတ်နေပါသည်..."):
            # ၁။ Image Processing ကို ပိုမိုပြတ်သားအောင်လုပ်ခြင်း
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Contrast ကို မြှင့်တင်ပေးခြင်း (CLAHE)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            processed_img = clahe.apply(gray)
            
            h, w = img.shape[:2]
            grid_data = [["" for _ in range(8)] for _ in range(num_rows)]
            # contrast_ths နှင့် adjust_contrast ထည့်သွင်းခြင်းဖြင့် အဖျော့စာလုံးများကို ပိုဖတ်နိုင်စေသည်
            results = reader.readtext(processed_img, detail=1, contrast_ths=0.1, adjust_contrast=0.6)

            # ၂။ Column Boundaries ကို လက်ရေးမူနှင့် အံဝင်ခွင်ကျညှိခြင်း
            # ပုံစံ (၈) တိုင်တွင် အကွက်အကျယ်များ မတူညီတတ်သဖြင့် ရာခိုင်နှုန်းကို ညှိထားသည်
            col_steps = [0.13, 0.24, 0.38, 0.49, 0.63, 0.74, 0.88, 1.0]

            for (bbox, text, prob) in results:
                # Bbox အလယ်မှတ်ကို ယူခြင်း
                cx = np.mean([p[0] for p in bbox])
                cy = np.mean([p[1] for p in bbox])
                
                rel_x = cx / w
                c_idx = 0
                for i, step in enumerate(col_steps):
                    if rel_x <= step:
                        c_idx = i
                        break
                
                r_idx = int((cy / h) * num_rows)

                if 0 <= r_idx < num_rows and 0 <= c_idx < 8:
                    txt = text.upper().strip()
                    
                    # ဂဏန်းတိုင်များအတွက် အထူးသန့်စင်ခြင်း
                    if c_idx % 2 == 0: 
                        # 'S' ကို '5', 'T' ကို '7' စသဖြင့် အမှားပြင်ခြင်း
                        repls = {'S': '5', 'T': '7', 'Z': '7', 'G': '6', 'I': '1', 'L': '1', 'O': '0', 'B': '8'}
                        for k, v in repls.items():
                            txt = txt.replace(k, v)
                        # ဂဏန်းနှင့် R ကလွဲပြီး အားလုံးဖယ်မည်
                        txt = re.sub(r'[^0-9R]', '', txt)
                        # ဂဏန်း ၃ လုံးထက်ကျော်နေပါက (ကပ်နေလျှင်) ရှေ့ ၃ လုံးကိုသာယူမည်
                        if len(txt) > 3 and 'R' not in txt:
                            txt = txt[:3]
                    
                    grid_data[r_idx][c_idx] = txt

            # ၃။ Ditto (ဒစ်တို) Logic ကို အစုံအလင်ထည့်သွင်းခြင်း
            for c in range(num_cols_active):
                last_val = ""
                for r in range(num_rows):
                    curr = str(grid_data[r][c]).strip()
                    
                    # လက်ရေးမူတွင် ditto ဖြစ်နိုင်သော ပုံစံများအားလုံး
                    is_ditto = curr in ['"', "''", "4", "LL", "Y", "V", "11", "U", "W", "-", "y"] or (not curr.isalnum() and curr != "")
                    
                    if (curr == "" or is_ditto) and last_val != "":
                        grid_data[r][c] = last_val
                    elif curr != "":
                        last_val = curr

            st.session_state['data_final'] = grid_data

# ---------------- ၇။ GOOGLE SHEET UPLOAD ----------------
if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး Google Sheet သို့ ပို့ရန်")
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Upload to Google Sheet"):
        try:
            # GCP Credentials
            secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
            secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
            client = gspread.authorize(creds)

            ss = client.open("LotteryData")
            sh1 = ss.get_worksheet(0) # Raw Data
            sh2 = ss.get_worksheet(1) # Sum Data

            # Sheet 1 သို့ ပို့ခြင်း
            sh1.append_rows(edited_data)

            # ပေါင်းခြင်း Logic
            master_sum = {}
            for row in edited_data:
                for i in range(0, num_cols_active, 2):
                    n_txt = str(row[i]).strip()
                    a_txt = str(row[i+1]).strip()
                    if n_txt and a_txt:
                        bet_res = process_ocr_results(n_txt, a_txt)
                        for g, val in bet_res.items():
                            master_sum[g] = master_sum.get(g, 0) + val

            # Sheet 2 သို့ အကျဉ်းချုပ်ပို့ခြင်း
            sh2.clear()
            final_list = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
            sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + final_list)

            st.success("🎉 Google Sheet သို့ အောင်မြင်စွာ ပို့ပြီးပါပြီ!")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")