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

# --- IMPORT HANDLING ---
try:
    from crypto_bot import (
        init_crypto_state, crypto_trading_loop,
        get_crypto_positions
    )
    CRYPTO_BOT_AVAILABLE = True
except ImportError:
    CRYPTO_BOT_AVAILABLE = False

# Try to import NSE
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
        /* ----- Global White Background ----- */
        /* Forces the main app container to be pure white */
        .stApp {
            background-color: #ffffff !important;
            color: #000000; /* Ensures text is black */
        }
        
        /* ----- Metric Box Styling ----- */
        /* Gives metric boxes a subtle off-white contrast so they don't disappear */
        div[data-testid="metric-container"] {
            background-color: #f9fafb; /* Very light gray for contrast */
            border: 1px solid #e5e7eb;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }

        /* ----- Dataframes ----- */
        /* Ensures dataframe backgrounds blend in */
        div[data-testid="stDataFrame"] {
            background-color: #ffffff;
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
AIROBOTS_URL = "https://airobots.streamlit.app/"
DB_PATH = "paper_trades.db"
CRYPTO_SYMBOLS_USDT = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]

# ---------------------------
# ROBUST DATA FETCHING
# ---------------------------
@st.cache_data(ttl=600) # Cache to prevent Rate Limits
def get_safe_crypto_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1mo")
        if history.empty: return None
        return history
    except:
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

for key in ["engine_status", "engine_running", "loop_started", 
            "crypto_running", "crypto_status", "crypto_loop_started"]:
    if key not in st.session_state:
        st.session_state[key] = False if "running" in key or "started" in key else "Idle"

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

def show_pnl_page():
    st.title("📊 PNL Log")
    st.write("PNL Data will appear here once trades execute.")

def show_crypto_page():
    st.title("🤖 Live Crypto Paper Trading Bot")
    # Refresh every 2 minutes to see live position updates
    st_autorefresh(interval=120_000, key="crypto_auto_refresh") 

    # 1. LIVE MARKET DATA
    st.subheader("📈 Real-Time Market Data")
    
    selected_crypto = st.selectbox("Select Asset", CRYPTO_SYMBOLS_USDT)
    
    # Safe Fetch
    data = get_safe_crypto_data(selected_crypto)
    
    if data is not None:
        curr_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2]
        delta = ((curr_price - prev_price)/prev_price)*100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Price", f"${curr_price:,.2f}", f"{delta:.2f}%")
        m2.metric("High", f"${data['High'].iloc[-1]:,.2f}")
        m3.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        # Simple Chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
        fig.update_layout(
            height=400, 
            margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor='rgba(0,0,0,0)', # Transparent plot background
            paper_bgcolor='rgba(0,0,0,0)', # Transparent paper background
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'), # Light gray grid
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Market data temporarily unavailable (Rate Limit). Retrying...")

    st.markdown("---")

    # 2. BOT CONTROLS (Simulated / Paper Trading)
    st.subheader("⚙️ Bot Controls")
    
    if not CRYPTO_BOT_AVAILABLE:
        st.error("⚠️ 'crypto_bot.py' is missing.")
        return

    col1, col2 = st.columns(2)
    status = st.session_state.get("crypto_status", "Idle")
    
    with col1:
        if st.button("▶️ Start Live Paper Trading"):
            st.session_state["crypto_running"] = True
            st.session_state["crypto_status"] = "Scanning Live Market..."
    with col2:
        if st.button("⏹ Stop Bot"):
            st.session_state["crypto_running"] = False
            st.session_state["crypto_status"] = "Stopped"

    st.info(f"Status: {status}")

    # 3. LIVE POSITIONS
    st.subheader("📍 Active Bot Positions")
    try:
        positions = get_crypto_positions() 
        if positions is not None and not positions.empty:
            st.dataframe(positions, use_container_width=True)
        else:
            st.caption("No active trades. The bot is scanning for volatility...")
    except Exception as e:
        st.error(f"Error loading positions: {e}")

# ---------------------------
# MAIN
# ---------------------------
def main():
    apply_custom_style()
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Bitcoin_logo.svg/1200px-Bitcoin_logo.svg.png", width=50)
    page = st.sidebar.radio("Navigation", ["Paper Trading", "PNL Log", "Crypto Bot"])
    
    # Start threads if not started
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
