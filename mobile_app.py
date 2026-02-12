import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# --- Business Logic: 1500*1000 parsing ---
def parse_bet_amount(amt_str):
    """
    1500*1000 သို့မဟုတ် 500500 လို့ ဖတ်မိတာကို ခွဲထုတ်ခြင်း
    """
    # OCR က * ကို 00 လို့ မှားဖတ်တတ်သဖြင့် 0 သုံးလုံးထက်ပိုပါက ခွဲထုတ်ကြည့်ခြင်း
    if len(amt_str) >= 4 and '00' in amt_str:
        parts = amt_str.split('00') # ဥပမာ 1500001000 -> ['15', '10']
        if len(parts) >= 2:
            main = int(parts[0] + "00") if parts[0] else 0
            back = int(parts[1]) if parts[1] else 0
            return main, back
    
    # ပုံမှန် * ပါခဲ့လျှင်
    if '*' in amt_str:
        p = amt_str.split('*')
        return int(p[0]) if p[0] else 0, int(p[1]) if p[1] else 0
        
    return int(amt_str) if amt_str.isdigit() else 0, 0

def get_r_list(num_str):
    """ဂဏန်းတစ်ခု၏ ပတ်လည် ၅ လုံးကို ရှာခြင်း (မူရင်းမပါ)"""
    if len(num_str) != 3: return []
    all_perms = sorted(list(set([''.join(p) for p in permutations(num_str)])))
    if num_str in all_perms:
        all_perms.remove(num_str)
    return all_perms

# --- App Logic ---
st.title("🎰 Lottery Pro: Advanced Betting Logic")

# (OCR Setup & Sidebar - Same as before)
with st.sidebar:
    st.header("⚙️ Settings")
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=50)
    col_mode = st.selectbox("အတိုင်အရေအတွက်", ["၂ တိုင်", "၄ တိုင်", "၆ တိုင်", "၈ တိုင်"], index=3)
    num_cols = int(col_mode.split()[0])

# ... (Image Upload & Reader - Same as before) ...

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန် (ဂဏန်း/ထိုးကြေး စစ်ပါ)")
    # Edit ပိုလုပ်ရလွယ်အောင် display လုပ်ထားပါတယ်
    edited_data = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("✅ Google Sheet သို့ ပို့မည် (Logic အသစ်ဖြင့်)"):
        client = get_gspread_client() # type: ignore
        ss = client.open("LotteryData")
        sh1, sh2, sh3 = ss.get_worksheet(0), ss.get_worksheet(1), ss.get_worksheet(2)
        
        sh1.append_rows(edited_data) # Raw save
        
        master_sum = {}
        voucher_data = []

        for row in edited_data:
            for i in range(0, 8, 2):
                num = str(row[i]).strip()
                bet_raw = str(row[i+1]).strip().replace(' ', '')
                
                if num and bet_raw:
                    # Logic: 1500*1000 ခွဲထုတ်ခြင်း
                    main_amt, total_r_amt = parse_bet_amount(bet_raw)
                    
                    # ၁။ မူရင်းဂဏန်းအတွက်ပေါင်းခြင်း
                    master_sum[num] = master_sum.get(num, 0) + main_amt
                    
                    # ၂။ ပတ်လည် (R) များအတွက် ခွဲဝေခြင်း
                    r_nums = get_r_list(num)
                    if r_nums and total_r_amt > 0:
                        each_r_amt = total_r_amt // len(r_nums)
                        for r_n in r_nums:
                            master_sum[r_n] = master_sum.get(r_n, 0) + each_r_amt
                    
                    # ၃။ ၃၀၀၀ ကျော်လျှင် Sheet 3 ပို့ရန် (Voucher)
                    if main_amt + total_r_amt > 3000:
                        voucher_data.append([num, (main_amt + total_r_amt) - 3000, "Limit Over"])

        # Update Sheet 2 (Aggregated)
        sh2.clear()
        final_sorted = [[k, master_sum[k]] for k in sorted(master_sum.keys())]
        sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + final_sorted)
        
        # Update Sheet 3
        sh3.append_rows(voucher_data)
        
        st.success("🎉 ပတ်လည်ခွဲဝေမှုနှင့် ပေါင်းလဒ်များ အားလုံး မှန်ကန်စွာ ပို့ဆောင်ပြီးပါပြီ။")