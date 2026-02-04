import streamlit as st
import numpy as np
import easyocr
import gspread
from PIL import Image
from oauth2client.service_account import ServiceAccountCredentials

st.set_page_config(page_title="Lottery Pro 2026", layout="wide")

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

reader = load_ocr()
st.title("🎰 Lottery OCR Pro")

# 1. Credentials (Raw string 'r' ကို သုံးပြီး PC Syntax Error ကို ဖြေရှင်းထားသည်)
creds_info = {
    "type": "service_account",
    "project_id": "turing-striker-485507-e9",
    "private_key_id": "2a3673384a5167d8776f40f8d77baed6e870d258",
    "private_key": r"-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCdhKlUPaVYSdyj\nIG7cc4mlQmked5u8PENmaSeRnZhGEBE3OvXsR5nS2pcLz0EdtACS/bebuyd47xVX\nzUX1n1w62YRSmHDjOf1a2c/F7yXPmvA39LeKthJPFqKaBbTTXADPT9N4meZcp7Kt\nTuiE4zr9XTVLvg30dr1zwTle3Sjv6UvDEcjTMnk3iT3KbFa8JoMF31Z5OZnjD3li\ngXR7pla/67HgQWvTuifDaOpPN2wUHWbBu+9wPTglp1sAje5KADzw0PJQXsD8RTQv\nD7925AmKX0NHNGrivbtFksL855WFW290e4BMbgOnydehG4PrO5Ih0NpDzazk+FSu\nCJMQ+JT/AgMBAAECggEAF3Vkgw4OkVL9X8fv6rdEO7njuNdcTft1mDYMirRQiLf8\n2wpWCfhx3APqNHvgY+GPp6nPIzYLVLMv17HTpkGzn8oO2al9P8Ho+D5G92CRyNTM\ny1s6eGx8K+N1l4BGY/x9hw0JiUT2QzybBhIh2888KkksK5m5hKZTDv8ATHSdvk4h\nkgAJVY7pEv3+mN6B3g5idZeYjql4CnfOk7A79LDNvUs0ZTrlRbesKnowt1AGw4qk\nh/tzrejw6AeZuPA9FrBqf3FF9U4qgYhgyaPmSJsMzYUjxxSjckJAsD95+xhc0u/a\nwHFefnWY4tf676fgJw7jBDDOTn5NNorxI+lY0gW4QQKBgQDQIaJWCzip4JwnEsbF\njvQAcE6Rhw6vzej/ogrj8aSMKXDbmkWZ2URCSBSEYPinwm3h+sEDGLZE0zwyUm/Q\nTO0CVydkeKZkPD+RPuUzsteYR5vCwScDP41F0tFwiH216gWNobU4BIneUOYPwkx3\nm+2VGuAHpt3oCjf07fFMHevPQQKBgQDBvwX2u5JhggcEqGnmBYKwA3oYQXztZdtS\n46MLmVyW5qDpilMY7M0PAvSGoAYVeFLPZUpcx+BM2wlueqGKoHpISEwKlOX75OGa\nOshwKxgFRQDex94eFKw/iww1c8eJAg+wLnEiv0ZVgrjXPumhw8PXo9JGUI8hnh5x\nseNit1+UPwKBgFQQIacrJRnH9In7lXkZwgejVLiGmjH7ss39PvTOFq3d1w17g/0d\nueojftXw7L2lVAhf5TFA++1UfA2/KhYx29CELw7vhBUcGHirtJtq8UU45vqEVSOE\uaSn/5u6JTwiZ1fLJoyXmK/IcQOQcJ4mxpDgp/evOBWOewdcS4d41lOBAoGBAKEq\nOdfIqDecZiIlxhGlu9SWz9WlhDomZI9K9LINXMvaBTi/6+fr85ftKWNjciwh7yC+\nbWFIkvjbq7jPIdmjLJU8LqUOv1EOT+xvwZQtBMo9YD/xmn8DS1WAYSOFsBH0OQCh\nYVM6MVOobgH/P3Fk22Bh0eTT1nxsf36sLy5Kw6MZAoGAEYHEmmfEASTj0kuIs/AY\nGlhd1ppF/T+hNnZ1XTsChrrj67SScqmOLDEcPCTSglr83M6hQEXFGerJVdE301//\ni7aKN9B6WKBFgObE0LXQXye+q8C4damqLd/nxXvyGUdMNIv5/dPAY/PtREvAort4\n4yMdiRqWEXQ8Y/ygC1unS2A=\n-----END PRIVATE KEY-----\n",
    "client_email": "mylotteryocr@turing-striker-485507-e9.iam.gserviceaccount.com",
    "client_id": "115799782494135939448"
}

# 2. Sidebar Settings (Default 8 တိုင်၊ 35 တန်း)
with st.sidebar:
    st.header("⚙️ Settings")
    num_cols = st.selectbox("တိုင်အရေအတွက်", [2, 4, 6, 8], index=3)
    num_rows = st.number_input("အတန်းအရေအတွက်", min_value=1, value=35)

uploaded_file = st.file_uploader("ပုံတင်ရန်", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    st.image(image, use_container_width=True)

    if st.button("🔍 AI ဖြင့် ဖတ်မည်"):
        with st.spinner("AI ဖတ်နေပါသည်..."):
            results = reader.readtext(img_array)
            h, w = img_array.shape[:2]
            grid_data = [["" for _ in range(num_cols)] for _ in range(num_rows)]
            
            y_pts = [res[0][0][1] for res in results] if results else [0]
            top_y = max(0, min(y_pts) - 10)
            cell_w, cell_h = w / num_cols, (h - top_y) / num_rows

            last_values = [""] * num_cols
            for (bbox, text, prob) in results:
                cx, cy = np.mean([p[0] for p in bbox]), np.mean([p[1] for p in bbox])
                c_idx, r_idx = int(cx // cell_w), int((cy - top_y) // cell_h)
                
                if 0 <= r_idx < num_rows and 0 <= c_idx < num_cols:
                    clean = text.strip().replace(" ", "")
                    # "။" သို့မဟုတ် တူညီသော သင်္ကေတများ တွေ့လျှင် Ditto အဖြစ် သတ်မှတ်
                    ditto_list = ["။", "||", "11", "II", "=", "—", "..", "::", "1/"]
                    if any(sym in clean for sym in ditto_list):
                        grid_data[r_idx][c_idx] = "DITTO_MARK"
                    else:
                        grid_data[r_idx][c_idx] = clean

            # Auto-fill Logic
            for r in range(num_rows):
                for c in range(num_cols):
                    if grid_data[r][c] == "DITTO_MARK":
                        grid_data[r][c] = last_values[c]
                    elif grid_data[r][c] != "":
                        last_values[c] = grid_data[r][c]
            st.session_state['data_final'] = grid_data

if 'data_final' in st.session_state:
    st.subheader("📝 စစ်ဆေးပြီး ပြင်ဆင်ရန်")
    edited_data = st.data_editor(st.session_state['data_final'], num_rows="dynamic")

    if st.button("✅ Google Sheet သို့ ပို့မည်"):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            # Formatting Key for Google Auth
            key = creds_info["private_key"].replace(r"\n", "\n")
            creds_info["private_key"] = key
            
            creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_info, scope)
            client = gspread.authorize(creds)
            sheet = client.open("LotteryData").sheet1
            sheet.clear()
            sheet.update("A1", edited_data)
            st.success("အောင်မြင်စွာ ပို့ဆောင်ပြီးပါပြီ။")
            st.balloons()
        except Exception as e:
            st.error(f"Error: {str(e)}")
