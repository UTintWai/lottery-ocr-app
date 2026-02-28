import streamlit as st
import numpy as np
import cv2
import easyocr
import re
import gspread
import gc
from oauth2client.service_account import ServiceAccountCredentials

# --- PAGE CONFIG ---
st.set_page_config(page_title="Lottery PC Max v42", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

def save_to_sheets_v42(data):
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds_dict = st.secrets["gcp_service_account"] 
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        sh = client.open("LotteryData")
        sheet = sh.get_worksheet(0)
        
        table_rows = []
        for row in data:
            if any(str(c).strip() for c in row[:8]):
                clean_row = [f"'{str(c)}" if str(c).strip() != "" else "" for c in row[:8]]
                table_rows.append(clean_row)
        
        if table_rows:
            sheet.append_rows(table_rows, value_input_option='USER_ENTERED')
            return True
        return False
    except Exception as e:
        st.error(f"Sheet Error: {str(e)}")
        return False

def process_v42(img):
    reader = load_ocr()
    h, w = img.shape[:2]
    target_w = 1600 # ပိုမိုကြည်လင်စေရန် Resolution ထပ်မြှင့်သည်
    img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # --- TRIPLE-PASS PROCESSING ---
    # မတူညီသော ပုံရိပ်အခြေအနေ ၃ မျိုးဖြင့် ဖတ်မည်
    # ၁။ မူရင်း၊ ၂။ အလင်းတင်ထားသောပုံ၊ ၃။ Contrast မြှင့်ထားသောပုံ
    passes = [
        gray,
        cv2.convertScaleAbs(gray, alpha=1.2, beta=10),
        cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    ]
    
    all_raw_results = []
    
    for p_img in passes:
        h_p = p_img.shape[0]
        num_segments = 6
        for i in range(num_segments):
            y1 = max(0, int(h_p * (i/num_segments)) - 30)
            y2 = min(h_p, int(h_p * ((i+1)/num_segments)) + 30)
            res = reader.readtext(p_img[y1:y2, :], paragraph=False)
            for (bbox, text, prob) in res:
                all_raw_results.append({
                    'x': np.mean([p[0] for p in bbox]),
                    'y': np.mean([p[1] for p in bbox]) + y1,
                    'text': text,
                    'prob': prob
                })
    
    if not all_raw_results: return []

    # ဒေတာထပ်နေသည်များကို ဖယ်ထုတ်ခြင်း (Deduplication)
    all_raw_results.sort(key=lambda k: (k['y'], k['x']))
    unique_results = []
    if all_raw_results:
        unique_results.append(all_raw_results[0])
        for i in range(1, len(all_raw_results)):
            # နေရာချင်းအရမ်းကပ်နေလျှင် တစ်ခုပဲယူမည်
            prev = unique_results[-1]
            curr = all_raw_results[i]
            if abs(curr['y'] - prev['y']) < 10 and abs(curr['x'] - prev['x']) < 50:
                if curr['prob'] > prev['prob']:
                    unique_results[-1] = curr
            else:
                unique_results.append(curr)

    # ROW GROUPING (စာကြောင်းခွဲခြင်း)
    unique_results.sort(key=lambda k: k['y'])
    rows = []
    curr_row = [unique_results[0]]
    for i in range(1, len(unique_results)):
        if unique_results[i]['y'] - curr_row[-1]['y'] < 25: 
            curr_row.append(unique_results[i])
        else:
            rows.append(curr_row)
            curr_row = [unique_results[i]]
    rows.append(curr_row)

    # --- 8-COLUMN TABLE MAPPING ---
    final_table = []
    col_width = target_w / 8
    
    for r_items in rows:
        row_cells = ["" for _ in range(8)]
        for item in r_items:
            c_idx = int(item['x'] // col_width)
            if 0 <= c_idx < 8:
                txt = re.sub(r'[^0-9"။=LVUYI/]', '', item['text'].upper())
                # အကွက်ထဲမှာ စာသားရှိပြီးသားဆိုလျှင် မထည့်တော့ပါ
                if not row_cells[c_idx]:
                    row_cells[c_idx] = txt.strip()
        
        for c in range(8):
            v = row_cells[c]
            if c % 2 == 0 and v.isdigit():
                row_cells[c] = v.zfill(3)[:3]
            elif any(m in v for m in ['"', '။', '=', 'L', 'V', 'U', 'Y', 'I', '/']):
                row_cells[c] = "DITTO"
        final_table.append(row_cells)

    # Smart Fill Down
    for c in [1, 3, 5, 7]:
        last_amt = ""
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO" and last_amt: final_table[r][c] = last_amt
            elif final_table[r][c].isdigit(): last_amt = final_table[r][c]
            
    for c in [0, 2, 4, 6]:
        for r in range(len(final_table)):
            if final_table[r][c] == "DITTO": final_table[r][c] = ""
            
    return final_table

# --- UI ---
st.title("🔢 PC Max-Accuracy Scanner v42")
st.write("Triple-Pass Scan စနစ်ဖြင့် စာလုံးများကို အသေအချာ ရှာဖွေဖတ်ပေးပါမည်။")

up_file = st.file_uploader("ဗောက်ချာပုံတင်ပါ", type=['jpg', 'jpeg', 'png'])

if up_file:
    file_bytes = np.frombuffer(up_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, width=800)
    
    if st.button("🔍 Run Max-Accuracy Scan"):
        with st.spinner("အသေးစိတ် ဖတ်နေပါသည်... ၃ ကြိမ် ပြန်လှန်စစ်ဆေးနေသဖြင့် ခဏစောင့်ပေးပါ..."):
            res = process_v42(img)
            st.session_state['data_v42'] = res

if 'data_v42' in st.session_state:
    st.subheader("စစ်ဆေးရန် ဇယား (A မှ H)")
    edited = st.data_editor(st.session_state['data_v42'], use_container_width=True)
    
    if st.button("💾 Save to Google Sheet"):
        if save_to_sheets_v42(edited):
            st.success("Google Sheet ထဲသို့ ဇယားအတိုင်း အောင်မြင်စွာ သိမ်းဆည်းပြီးပါပြီ!")
