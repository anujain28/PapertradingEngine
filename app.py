import os
import time
import threading
import datetime as dt
import sqlite3
from typing import Dict, List, Optional
import random

import configparser
import pytz
import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from telegram.ext import Application
import yfinance as yf
import plotly.graph_objects as go

# --- IMPORT HANDLING (Prevents crash if crypto_bot is missing) ---
try:
    from crypto_bot import (
        init_crypto_state, get_binance_client, crypto_trading_loop,
        save_binance_config, load_binance_config, get_crypto_positions,
        get_crypto_trades
    )
    CRYPTO_BOT_AVAILABLE = True
except ImportError:
    CRYPTO_BOT_AVAILABLE = False

# Try to import NSE, fallback if fails
try:
    from nsepython import nse_quote_ltp
    NSE_AVAILABLE = True
except ImportError:
    NSE_AVAILABLE = False

# ---------------------------
# PAGE CONFIG + GLOBAL STYLE
# ---------------------------
st.set_page_config(page_title="AI Paper Trading", layout="wide", page_icon="📈")

def apply_custom_style():
    st.markdown("""
        <style>
        .stApp { color: #000000; }
        div[data-testid="metric-container"] {
            background-color: #f0f2f6;
            border: 1px solid #d6d6d6;
            padding: 10px;
            border-radius: 10px;
        }
        </style>
        """, unsafe_allow_html=True)

# ---------------------------
# CONFIG
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")
TRADING_START = dt.time(9, 30)
TRADING_END = dt.time(15, 30)
START_CAPITAL = 100000.0
MAX_UTILIZATION = 0.60
AIROBOTS_URL = "https://airobots.streamlit.app/"
DB_PATH = "paper_trades.db"
CONFIG_PATH = "telegram_config.ini"
CRYPTO_SYMBOLS_USDT = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]

# ---------------------------
# ROBUST DATA FETCHING (THE FIX)
# ---------------------------
@st.cache_data(ttl=600) # Cache data for 10 minutes to prevent Rate Limit Errors
def get_safe_crypto_data(symbol):
    """Safely fetch crypto data with error handling"""
    try:
        # Tries to fetch 1 month of history
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1mo")
        if history.empty:
            return None
        return history
    except Exception as e:
        # If API fails, return None (don't crash the app)
        return None

def get_market_price(symbol: str) -> Optional[float]:
    """Fallback price fetcher"""
    try:
        df = get_safe_crypto_data(symbol)
        if df is not None:
            return df["Close"].iloc[-1]
        return None
    except:
        return None

def get_ltp(symbol: str) -> Optional[float]:
    """Get stock price safely"""
    if NSE_AVAILABLE:
        try:
            val = nse_quote_ltp(symbol)
            if val: return float(val)
        except:
            pass
    # Fallback to Yfinance
    try:
        yf_sym = symbol + ".NS"
        data = get_safe_crypto_data(yf_sym)
        if data is not None:
            return data["Close"].iloc[-1]
    except:
        pass
    return None

# ---------------------------
# STATE MANAGEMENT
# ---------------------------
if "state" not in st.session_state:
    st.session_state["state"] = {
        "capital": START_CAPITAL,
        "equity": START_CAPITAL,
        "positions": {},
    }

# Initialize flags
for key in ["engine_status", "engine_running", "loop_started", "telegram_started", 
            "crypto_running", "crypto_status", "crypto_loop_started"]:
    if key not in st.session_state:
        st.session_state[key] = False if "running" in key or "started" in key else "Idle"

if "report_time" not in st.session_state:
    st.session_state["report_time"] = dt.time(16, 0)
if "last_top5" not in st.session_state:
    st.session_state["last_top5"] = []

if CRYPTO_BOT_AVAILABLE:
    init_crypto_state()

# ---------------------------
# DATABASE
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS equity_snapshots (id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, timestamp TEXT, equity REAL, capital REAL)""")
    conn.commit()
    conn.close()
init_db()

# ---------------------------
# UI PAGES
# ---------------------------
def show_paper_trading_page():
    st.title("📈 AI Paper Trading Engine")
    st_autorefresh(interval=120_000, key="auto_refresh")
    
    state = st.session_state["state"]
    col1, col2 = st.columns(2)
    col1.metric("Free Capital", f"₹{state['capital']:,.2f}")
    col2.metric("Equity", f"₹{state['equity']:,.2f}")
    
    st.info(f"Engine Status: {st.session_state.get('engine_status')}")
    st.caption("Positions will appear here when the engine trades.")

def show_pnl_page():
    st.title("📊 PNL Log")
    st.write("PNL Data will appear here once trades execute.")

def show_crypto_page():
    st.title("🤖 Crypto Grid Trading Bot")
    # Increase refresh time to 5 mins to save API calls
    st_autorefresh(interval=300_000, key="crypto_auto_refresh") 

    # 1. LIVE DASHBOARD (Visuals)
    st.subheader("📈 Live Market Overview")
    
    selected_crypto = st.selectbox("Select Asset to View", CRYPTO_SYMBOLS_USDT)
    
    # --- SAFE DATA FETCHING ---
    data = get_safe_crypto_data(selected_crypto)
    
    if data is not None:
        curr_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        delta = ((curr_price - prev_price)/prev_price)*100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"${curr_price:,.2f}", f"{delta:.2f}%")
        m2.metric("24h High", f"${data['High'].iloc[-1]:,.2f}")
        m3.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        # Chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
        fig.update_layout(height=400, margin=dict(l=0,r=0,t=0,b=0))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"⚠️ Could not fetch live data for {selected_crypto}. The API rate limit may have been reached. Try again in a few minutes.")

    st.markdown("---")

    # 2. BOT CONTROLS
    st.subheader("⚙️ Bot Configuration")
    
    if not CRYPTO_BOT_AVAILABLE:
        st.error("Crypto Bot module not found. Please create 'crypto_bot.py'.")
        return

    # API Keys
    with st.expander("API Keys (Binance)", expanded=False):
        c1, c2 = st.columns(2)
        api = c1.text_input("API Key", type="password")
        sec = c2.text_input("Secret", type="password")
        if st.button("Save Keys"):
            save_binance_config(api, sec)
            st.success("Keys Saved!")

    # Controls
    col1, col2 = st.columns(2)
    status = st.session_state.get("crypto_status", "Idle")
    
    with col1:
        if st.button("▶️ Start Bot"):
            st.session_state["crypto_running"] = True
            st.session_state["crypto_status"] = "Running"
    with col2:
        if st.button("⏹ Stop Bot"):
            st.session_state["crypto_running"] = False
            st.session_state["crypto_status"] = "Stopped"

    st.info(f"Status: {status}")

    # 3. LIVE POSITIONS
    st.subheader("📍 Active Grid Positions")
    try:
        positions = get_crypto_positions() 
        if positions is not None and not positions.empty:
            st.dataframe(positions, use_container_width=True)
        else:
            st.caption("No active positions.")
    except Exception as e:
        st.error(f"Error loading positions: {e}")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Bitcoin_logo.svg/1200px-Bitcoin_logo.svg.png", width=50)
    page = st.sidebar.radio("Navigation", ["Paper Trading", "PNL Log", "Crypto Bot"])

    if not st.session_state.get("loop_started", False):
        st.session_state["loop_started"] = True
    
    if CRYPTO_BOT_AVAILABLE and not st.session_state.get("crypto_loop_started", False):
        t_crypto = threading.Thread(target=crypto_trading_loop, daemon=True)
        t_crypto.start()
        st.session_state["crypto_loop_started"] = True

    if page == "Paper Trading":
        show_paper_trading_page()
    elif page == "PNL Log":
        show_pnl_page()
    elif page == "Crypto Bot":
        show_crypto_page()

if __name__ == "__main__":
    main()
