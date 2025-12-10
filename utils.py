import streamlit as st
import requests
import sqlite3
import yfinance as yf
import pytz

# --- CONFIG ---
IST = pytz.timezone("Asia/Kolkata")
DB_PATH = "paper_trades.db"

# --- STYLING ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* =========================================
           1. GLOBAL THEME (White Background, Black Text)
           ========================================= */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Force all standard text to black */
        p, h1, h2, h3, h4, h5, h6, li, span, label, div {
            color: #000000 !important;
        }

        /* =========================================
           2. SIDEBAR (Dark Theme - Black BG, White Text)
           ========================================= */
        section[data-testid="stSidebar"] {
            background-color: #262730 !important;
        }
        
        /* Force ALL text in sidebar to be White */
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* --- FIX: SIDEBAR INPUTS (Black Box, White Text) --- */
        /* This targets the specific input container in the sidebar */
        section[data-testid="stSidebar"] input.st-be {
            background-color: #000000 !important;
            color: #ffffff !important;
            caret-color: #ffffff !important; /* White Cursor */
            border: 1px solid #555 !important;
        }
        /* Fallback for other input types */
        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            background-color: #000000 !important;
            border-color: #555 !important;
        }
        section[data-testid="stSidebar"] input {
            color: #ffffff !important; 
            background-color: transparent !important;
        }

        /* SIDEBAR EXPANDERS */
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[role="group"] {
            background-color: #262730 !important;
        }

        /* =========================================
           3. MAIN PAGE COMPONENTS
           ========================================= */
        
        /* METRIC BOXES (Light Grey) */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
            color: #000000 !important;
        }
        div[data-testid="metric-container"] label { color: #000000 !important; }

        /* TABLES & DATAFRAMES (White BG, Black Text) */
        div[data-testid="stDataFrame"] { background-color: #ffffff !important; }
        div[data-testid="stDataFrame"] div[class*="stDataFrame"] { color: #000000 !important; }
        div[data-testid="stTable"] { color: #000000 !important; }

        /* MAIN PAGE EXPANDERS (White BG, Black Text) */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] {
            background-color: #ffffff !important;
        }
        
        /* DROPDOWNS & SELECTBOXES (Main Page) */
        .main div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ccc;
        }
        .main div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        .main div[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* BUTTONS */
        .stButton > button {
            background-color: #e5e7eb !important;
            color: #000000 !important;
            border: 1px solid #9ca3af !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --- SIDEBAR CONFIG UI ---
def show_sidebar_config():
    st.sidebar.markdown("---")
    
    # 1. Telegram
    with st.sidebar.expander("📢 Telegram Alerts"):
        st.caption("Enter Bot Token & Chat ID")
        tg_token = st.text_input("Bot Token", value=st.session_state.get("tg_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))
        if st.button("💾 Save Telegram"):
            st.session_state["tg_token"] = tg_token
            st.session_state["tg_chat_id"] = tg_chat
            st.success("Saved!")

    # 2. Binance
    with st.sidebar.expander("🔌 Binance Keys"):
        st.caption("For Live Crypto Trading")
        api = st.text_input("API Key", value=st.session_state.get("binance_api", ""), type="password")
        sec = st.text_input("Secret Key", value=st.session_state.get("binance_secret", ""), type="password")
        if st.button("💾 Save Binance"):
            st.session_state["binance_api"] = api
            st.session_state["binance_secret"] = sec
            st.success("Saved!")

    # 3. Dhan
    with st.sidebar.expander("🇮🇳 Dhan Config"):
        st.caption("For Live Stock Trading")
        d_id = st.text_input("Client ID", value=st.session_state.get("dhan_client_id", ""))
        d_token = st.text_input("Access Token", value=st.session_state.get("dhan_token", ""), type="password")
        if st.button("💾 Save Dhan"):
            st.session_state["dhan_client_id"] = d_id
            st.session_state["dhan_token"] = d_token
            st.success("Saved!")

# --- HELPERS ---
def send_telegram_alert(message):
    token = st.session_state.get("tg_token")
    chat_id = st.session_state.get("tg_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        except: pass

@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        return yf.Ticker("INR=X").history(period="1d")["Close"].iloc[-1]
    except: return 84.0 

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()
