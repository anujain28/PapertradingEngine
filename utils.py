import streamlit as st
import pytz
import sqlite3
import requests
import yfinance as yf

# --- CONSTANTS (DEFINED FIRST) ---
IST = pytz.timezone("Asia/Kolkata")
DB_PATH = "paper_trades.db"

# --- TELEGRAM HELPER ---
def send_telegram_alert(message):
    token = st.session_state.get("tg_token")
    chat_id = st.session_state.get("tg_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        except: pass

# --- DATA HELPER ---
@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        return yf.Ticker("INR=X").history(period="1d")["Close"].iloc[-1]
    except: return 84.0

# --- DATABASE ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()

# --- STYLING ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* Global Reset */
        .stApp { background-color: #ffffff !important; color: #000000 !important; }
        p, h1, h2, h3, h4, h5, h6, li, span, label, div { color: #000000 !important; }

        /* Sidebar (Dark) */
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        
        /* Sidebar Inputs */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; color: #ffffff !important; caret-color: white !important; border: 1px solid #555 !important;
        }
        
        /* Metric Boxes */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa !important; border: 1px solid #dee2e6; color: #000000 !important; padding: 10px; border-radius: 8px;
        }
        div[data-testid="metric-container"] label { color: #000000 !important; }

        /* Tables */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] { background-color: #ffffff !important; color: #000000 !important; }
        
        /* Dropdowns & Selects (Main Page) */
        .main div[data-baseweb="select"] > div { background-color: #ffffff !important; color: #000000 !important; border: 1px solid #ccc; }
        .main div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #ffffff !important; }
        .main div[role="option"] { background-color: #ffffff !important; color: #000000 !important; }
        .main div[role="option"]:hover { background-color: #f0f2f6 !important; }

        /* Expanders */
        .main div[data-testid="stExpander"] details summary { background-color: #f8f9fa !important; color: #000000 !important; border: 1px solid #ddd; }
        .main div[data-testid="stExpander"] div[role="group"] { background-color: #ffffff !important; color: #000000 !important; }
        
        /* Buttons */
        .stButton > button { background-color: #e5e7eb !important; color: #000000 !important; border: 1px solid #ccc !important; }
        </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR CONFIG UI ---
def show_sidebar_config():
    st.sidebar.markdown("---")
    with st.sidebar.expander("📢 Telegram Alerts"):
        t1 = st.text_input("Bot Token", value=st.session_state.get("tg_token", ""), type="password")
        t2 = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))
        if st.button("💾 Save Telegram"):
            st.session_state["tg_token"] = t1
            st.session_state["tg_chat_id"] = t2
            st.success("Saved!")

    with st.sidebar.expander("🔌 Binance Keys"):
        k1 = st.text_input("API Key", type="password")
        k2 = st.text_input("Secret Key", type="password")
        if st.button("💾 Save Binance"): st.success("Saved!")

    with st.sidebar.expander("🇮🇳 Dhan Config"):
        d1 = st.text_input("Client ID")
        d2 = st.text_input("Access Token", type="password")
        if st.button("💾 Save Dhan"): st.success("Saved!")
