import streamlit as st
import requests
import yfinance as yf
import pytz
import sqlite3

# --- CONFIG ---
IST = pytz.timezone("Asia/Kolkata")
DB_PATH = "paper_trades.db"

# --- STYLING ---
def apply_custom_style():
    st.markdown("""
        <style>
        /* Global Styles */
        .stApp { background-color: #ffffff !important; color: #000000 !important; }
        
        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        
        /* Sidebar Inputs (Black BG, White Text) */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            caret-color: #ffffff !important;
            border: 1px solid #555 !important;
        }
        
        /* Metrics & Containers */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Expander & Table Fixes */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important; color: #000000 !important; border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] { background-color: #ffffff !important; }
        .main div[data-testid="stExpander"] table, td, th { color: #000000 !important; }
        
        /* Dropdowns */
        div[data-baseweb="select"] > div { background-color: #ffffff !important; color: black !important; }
        div[role="option"] { background-color: #ffffff !important; color: black !important; }
        </style>
        """, unsafe_allow_html=True)

# --- TELEGRAM ---
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
