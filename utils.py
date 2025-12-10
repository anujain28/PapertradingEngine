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
           1. GLOBAL MAIN PAGE (Pure White Theme)
           ========================================= */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .main p, .main h1, .main h2, .main h3, .main label, .main span, .main div {
            color: #000000 !important;
        }

        /* =========================================
           2. SIDEBAR (Dark Theme)
           ========================================= */
        section[data-testid="stSidebar"] {
            background-color: #262730 !important;
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* SIDEBAR INPUTS: Black Box, White Text, White Cursor */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            caret-color: #ffffff !important;
            border: 1px solid #666 !important;
        }
        section[data-testid="stSidebar"] label {
            color: #ffffff !important;
        }
        
        /* SIDEBAR EXPANDERS: Dark Background, White Text */
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
            background-color: #333333 !important;
            color: #ffffff !important;
            border: 1px solid #555;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary:hover {
            color: #ffbd45 !important; /* Gold hover effect */
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[role="group"] {
            background-color: #262730 !important;
            color: #ffffff !important;
        }

        /* =========================================
           3. MAIN PAGE ELEMENTS (Force White/Black)
           ========================================= */
        
        /* METRIC BOXES */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            color: #000000 !important;
            border-radius: 8px;
            padding: 10px;
        }
        div[data-testid="metric-container"] label { color: #000000 !important; }

        /* TABLES & DATAFRAMES */
        div[data-testid="stDataFrame"], 
        div[data-testid="stTable"], 
        div[class*="stDataFrame"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* EXPANDERS (Main Page Only - e.g. Live Grid Orders) */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* DROPDOWNS */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ccc;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        div[role="option"] {
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

# --- SIDEBAR CONFIGURATION UI ---
def show_sidebar_config():
    """Renders the configuration expanders in the sidebar."""
    st.sidebar.markdown("---")
    
    # 1. Telegram
    with st.sidebar.expander("📢 Telegram Alerts"):
        tg_token = st.text_input("Bot Token", value=st.session_state.get("tg_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))
        if st.button("💾 Save Telegram"):
            st.session_state["tg_token"] = tg_token
            st.session_state["tg_chat_id"] = tg_chat
            st.success("Saved!")

    # 2. Binance
    with st.sidebar.expander("🔌 Binance Keys"):
        api = st.text_input("API Key", value=st.session_state.get("binance_api", "") or "", type="password")
        sec = st.text_input("Secret Key", value=st.session_state.get("binance_secret", "") or "", type="password")
        if st.button("💾 Save Binance"):
            st.session_state["binance_api"] = api
            st.session_state["binance_secret"] = sec
            st.success("Saved!")

    # 3. Dhan
    with st.sidebar.expander("🇮🇳 Dhan Config"):
        d_id = st.text_input("Client ID", value=st.session_state.get("dhan_client_id", ""))
        d_token = st.text_input("Access Token", value=st.session_state.get("dhan_token", ""), type="password")
        if st.button("💾 Save Dhan"):
            st.session_state["dhan_client_id"] = d_id
            st.session_state["dhan_token"] = d_token
            st.success("Saved!")

# --- TELEGRAM HELPER ---
def send_telegram_alert(message):
    token = st.session_state.get("tg_token")
    chat_id = st.session_state.get("tg_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        except Exception as e:
            print(f"Telegram Error: {e}")

# --- DATA HELPERS ---
@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return 84.0 

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()
