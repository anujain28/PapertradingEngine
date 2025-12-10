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
        /* Global & Sidebar */
        .stApp { background-color: #ffffff !important; color: #000000 !important; }
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        
        /* Sidebar Inputs: Black BG, White Text, White Cursor */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            caret-color: #ffffff !important;
            border: 1px solid #555 !important;
        }
        
        /* Metric Boxes */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
            color: #000000 !important;
        }
        
        /* Dropdowns & Popovers (White BG, Black Text) */
        div[data-baseweb="select"] > div { background-color: #ffffff !important; color: black !important; }
        div[data-baseweb="popover"], div[data-baseweb="menu"] { background-color: #ffffff !important; }
        div[role="option"] { background-color: #ffffff !important; color: black !important; }
        div[role="option"]:hover { background-color: #f0f2f6 !important; }
        
        /* Expander Headers (Main Page) */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important; color: #000000 !important; border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] { background-color: #ffffff !important; }
        .main div[data-testid="stExpander"] table, td, th { color: #000000 !important; }
        
        /* Sidebar Expanders (Keep Dark) */
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
            background-color: #333 !important; color: white !important;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #e5e7eb !important; color: black !important; border: 1px solid #9ca3af !important;
        }
        </style>
        """, unsafe_allow_html=True)

# --- SIDEBAR CONFIG UI (Crucial Function) ---
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
        k1 = st.text_input("API Key", value=st.session_state.get("binance_api", ""), type="password")
        k2 = st.text_input("Secret Key", value=st.session_state.get("binance_secret", ""), type="password")
        if st.button("💾 Save Binance"):
            st.session_state["binance_api"] = k1
            st.session_state["binance_secret"] = k2
            st.success("Saved!")

    with st.sidebar.expander("🇮🇳 Dhan Config"):
        d1 = st.text_input("Client ID", value=st.session_state.get("dhan_client_id", ""))
        d2 = st.text_input("Access Token", value=st.session_state.get("dhan_token", ""), type="password")
        if st.button("💾 Save Dhan"):
            st.session_state["dhan_client_id"] = d1
            st.session_state["dhan_token"] = d2
            st.success("Saved!")

# --- TELEGRAM HELPER ---
def send_telegram_alert(message):
    token = st.session_state.get("tg_token")
    chat_id = st.session_state.get("tg_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        except: pass

# --- DATA HELPERS ---
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
