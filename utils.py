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
           1. GLOBAL RESET (White Theme)
           ========================================= */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        p, h1, h2, h3, h4, h5, h6, li, span, label, div {
            color: #000000 !important;
        }

        /* =========================================
           2. SIDEBAR (Dark Theme - Black/White)
           ========================================= */
        section[data-testid="stSidebar"] {
            background-color: #262730 !important;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        
        /* Sidebar Inputs: Black Box, White Text */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            caret-color: #ffffff !important;
            border: 1px solid #555 !important;
        }
        section[data-testid="stSidebar"] div[data-baseweb="input"] {
            background-color: #000000 !important;
        }
        
        /* Sidebar Expanders */
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
            background-color: #333 !important;
            color: #ffffff !important;
            border: 1px solid #555;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[role="group"] {
            background-color: #262730 !important;
        }

        /* =========================================
           3. MAIN PAGE INPUTS (White Box, Black Text)
           ========================================= */
        
        /* Number Inputs & Text Inputs (Main Page Only) */
        .main div[data-baseweb="input"] {
            background-color: #ffffff !important;
            border: 1px solid #ced4da !important;
        }
        .main input {
            color: #000000 !important;
        }
        
        /* Select Boxes (Main Page Only) */
        .main div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da !important;
        }
        .main div[data-baseweb="select"] span {
            color: #000000 !important;
        }
        
        /* Dropdown Popups (The List) */
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        div[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        div[role="option"]:hover {
            background-color: #f0f2f6 !important;
        }

        /* =========================================
           4. TABLES, EXPANDERS & METRICS
           ========================================= */
        
        /* Metric Boxes */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa !important;
            border: 1px solid #dee2e6;
            color: #000000 !important;
            border-radius: 8px;
            padding: 10px;
        }
        div[data-testid="metric-container"] label { color: #000000 !important; }

        /* Tables */
        div[data-testid="stDataFrame"], div[data-testid="stTable"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Main Page Expanders */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] {
            background-color: #ffffff !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #e5e7eb !important;
            color: #000000 !important;
            border: 1px solid #ccc !important;
        }
        .stButton > button:hover {
            background-color: #d1d5db !important;
            border-color: #000 !important;
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
        return yf.Ticker("INR=X").history(period="1d")["Close"].iloc[-1]
    except: return 84.0 

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()
