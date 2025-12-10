import streamlit as st
import requests
import sqlite3
import yfinance as yf
import pytz

# --- CONFIG ---
IST = pytz.timezone("Asia/Kolkata")
DB_PATH = "paper_trades.db"

# --- SIDEBAR CONFIG UI ---
def show_sidebar_config():
    st.sidebar.markdown("---")
    with st.sidebar.expander("📢 Telegram Alerts"):
        t1 = st.text_input("Bot Token", value=st.session_state.get("tg_token", ""), type="password")
        t2 = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))
        if st.button("💾 Save Telegram"):
            st.session_state["tg_token"] = t1; st.session_state["tg_chat_id"] = t2; st.success("Saved!")

    with st.sidebar.expander("🔌 Binance Keys"):
        st.text_input("API Key", type="password")
        st.text_input("Secret Key", type="password")
        if st.button("💾 Save Binance"): st.success("Saved!")

    with st.sidebar.expander("🇮🇳 Dhan Config"):
        st.text_input("Client ID")
        st.text_input("Access Token", type="password")
        if st.button("💾 Save Dhan"): st.success("Saved!")

# --- SHARED TOOLS ---
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
    try: return yf.Ticker("INR=X").history(period="1d")["Close"].iloc[-1]
    except: return 84.0 

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()

def apply_custom_style():
    # Only minimal global sidebar resets here. Page specific styles in page files.
    st.markdown("""<style>
        .stApp { background-color: #ffffff; color: black; }
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        section[data-testid="stSidebar"] input { background-color: #000; color: #fff; border: 1px solid #555; }
        </style>""", unsafe_allow_html=True)
