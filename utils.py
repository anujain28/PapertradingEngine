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
        
        /* Sidebar Inputs */
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
        div[data-testid="metric-container"] label { color: #000000 !important; }
        
        /* --- DROPDOWN & SELECTBOX FIX (CRITICAL) --- */
        /* This ensures the selection box itself is white with black text */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da !important;
        }
        /* This ensures the text inside the selection box is black */
        div[data-baseweb="select"] span {
            color: #000000 !important;
        }
        /* This targets the dropdown popup list */
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        /* This targets the options inside the list */
        div[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        /* Hover state for options */
        div[role="option"]:hover {
            background-color: #f0f2f6 !important;
        }
        
        /* --- TABLE & EXPANDER FIX --- */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Main Page Expanders (White BG, Black Text) */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] {
            background-color: #ffffff !important;
        }
        .main div[data-testid="stExpander"] p, 
        .main div[data-testid="stExpander"] span,
        .main div[data-testid="stExpander"] div {
            color: #000000 !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #e5e7eb !important; color: black !important; border: 1px solid #9ca3af !important;
        }
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
        except: pass

# --- DATA HELPERS ---
@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        if not data.empty: return data["Close"].iloc[-1]
    except: pass
    return 84.0 

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()
