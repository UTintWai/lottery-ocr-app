import streamlit as st
import numpy as np
import easyocr
import cv2
import re
import json
import gspread
from itertools import permutations
from oauth2client.service_account import ServiceAccountCredentials

# --- Google Sheets Setup ---
def get_gspread_client():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    secret_info = json.loads(st.secrets["GCP_SERVICE_ACCOUNT_FILE"])
    secret_info["private_key"] = secret_info["private_key"].replace("\\n", "\n")
    creds = ServiceAccountCredentials.from_json_keyfile_dict(secret_info, scope)
    return gspread.authorize(creds)

# --- Business Logic: Formatting & Ditto ---
def clean_and_format_data(grid, num_rows, num_cols):
    for c in range(num_cols):
        last_val = ""
        for r in range(num_rows):
            curr = str(grid[r][c]).strip().upper()
            
            # Ditto characters check
            is_ditto = any(s in curr for s in ["\"", "||", "1", "U", "''", "။", "〃", "=", "-", "U"])
            
            # Special Correction for specific errors mentioned
            if c % 2 != 0: # ထိုးကြေးတိုင်များအတွက်
                if curr == "3" and "36" in last_val: curr = "36"
                if curr == "2" and "24" in last_val: curr = "24"
                if curr == "0" and last_val != "": curr = last_val
                if curr == "1" and "24" in last_val: curr = "24"

            if (is_ditto or curr == "") and last_val != "":
                grid[r][c] = last_val
            elif curr != "":
                if c % 2 == 0: # ဂဏန်းတိုင် (1, 3, 5, 7)
                    val = re.sub(r'\D', '', curr)
                    if val:
                        formatted = val[-3:].zfill(3)
                        grid[r][c] = formatted
                        last_val = formatted
                else: # ထိုးကြေးတိုင် (2, 4, 6, 8)
                    val = re.sub(r'\D', '', curr)
                    grid[r][c] = val
                    last_val = val
    return grid

# --- Streamlit UI ---
st.title("🎰 Lottery Pro: Advanced Sorting & Voucher System")

# ... (OCR Loading & File Upload logic remains same as previous) ...

if 'data_final' in st.session_state:
    edited_df = st.data_editor(st.session_state['data_final'], use_container_width=True)

    if st.button("🚀 Process Sheets (1, 2 & 3)"):
        client = get_gspread_client()
        ss = client.open("LotteryData")
        
        # Sheet 1: Raw Data
        sh1 = ss.get_worksheet(0)
        sh1.append_rows(edited_df)

        # Sheet 2 & 3 Logic
        master_dict = {}
        voucher_list = [] # For Sheet 3
        
        # Process data for summing and voucher
        for row in edited_df:
            for i in range(0, 8, 2):
                num, amt_str = str(row[i]), str(row[i+1])
                if num and amt_str.isdigit():
                    amt = int(amt_str)
                    master_dict[num] = master_dict.get(num, 0) + amt
                    
                    # 💡 ထိုးကြေး ၃၀၀၀ ကျော်လျှင် Sheet 3 အတွက် သိမ်းဆည်းခြင်း
                    if amt > 3000:
                        voucher_list.append([num, amt - 3000, "Limit Exceeded"])

        # Update Sheet 2 (Sorted Sums)
        sh2 = ss.get_worksheet(1)
        sh2.clear()
        sorted_data = [[k, master_dict[k]] for k in sorted(master_dict.keys())]
        sh2.append_rows([["ဂဏန်း", "စုစုပေါင်း"]] + sorted_data)

        # Update Sheet 3 (Voucher / Over Limit)
        try:
            sh3 = ss.get_worksheet(2)
        except:
            sh3 = ss.add_worksheet(title="Sheet3", rows="100", cols="10")
        
        sh3.clear()
        sh3.append_rows([["ဂဏန်း", "ပိုငွေ", "မှတ်ချက်"]] + voucher_list)

        st.success("✅ Sheet 1 (Raw), Sheet 2 (Total), Sheet 3 (Voucher) အားလုံး အောင်မြင်စွာ ပို့ပြီးပါပြီ။")