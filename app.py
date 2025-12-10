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
import numpy as np
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
        /* Global Styles */
        .stApp { background-color: #ffffff !important; color: #000000 !important; }
        p, h1, h2, h3, h4, h5, h6, span, div, label, li { color: #000000 !important; }
        
        /* Sidebar */
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        
        /* Metrics & Containers */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            border-radius: 8px;
            padding: 10px;
        }
        
        /* Buttons */
        .stButton > button {
            background-color: #e5e7eb !important;
            color: black !important;
            border: 1px solid #9ca3af !important;
        }
        .stButton > button:hover {
            background-color: #d1d5db !important;
        }

        /* --- DROPDOWN MENU FIX (White BG, Black Text) --- */
        /* The container of the selected value */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da;
        }
        /* The dropdown popup */
        div[data-baseweb="popover"], div[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        /* The options inside */
        div[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        /* Hover state */
        div[role="option"]:hover {
            background-color: #f0f2f6 !important;
        }
        div[data-baseweb="select"] span {
            color: #000000 !important; 
        }
        
        /* Chart Override */
        .js-plotly-plot .plotly .modebar { display: none !important; }
        </style>
        """, unsafe_allow_html=True)

# ---------------------------
# CONFIG
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")
START_CAPITAL = 100000.0
DB_PATH = "paper_trades.db"

# We trade in USDT (USD) for the bot logic
CRYPTO_SYMBOLS_USD = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]

# ---------------------------
# DATA FETCHING
# ---------------------------
@st.cache_data(ttl=300) 
def get_safe_crypto_data(symbol, period="1mo"):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period=period)
        if history.empty: return None
        return history
    except:
        return None

def get_current_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return 0.0

@st.cache_data(ttl=3600) # Cache exchange rate for 1 hour
def get_usd_inr_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return 84.0 # Fallback rate if API fails

# ---------------------------
# STATE MANAGEMENT
# ---------------------------
if "state" not in st.session_state:
    st.session_state["state"] = {
        "capital": START_CAPITAL,
        "equity": START_CAPITAL,
        "positions": {},
    }

if "grid_bot_active" not in st.session_state:
    st.session_state["grid_bot_active"] = {} 

if "usd_inr" not in st.session_state:
    st.session_state["usd_inr"] = get_usd_inr_rate()

for key in ["engine_status", "engine_running", "loop_started", 
            "crypto_running", "crypto_status", "crypto_loop_started"]:
    if key not in st.session_state:
        st.session_state[key] = False if "running" in key or "started" in key else "Idle"

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
# PAGE 1: PAPER TRADING
# ---------------------------
def show_paper_trading_page():
    st.title("📈 AI Paper Trading Engine")
    st_autorefresh(interval=120_000, key="auto_refresh")
    state = st.session_state["state"]
    col1, col2 = st.columns(2)
    col1.metric("Free Capital", f"₹{state['capital']:,.2f}")
    col2.metric("Equity", f"₹{state['equity']:,.2f}")
    st.info(f"Engine Status: {st.session_state.get('engine_status')}")

# ---------------------------
# PAGE 2: PNL LOG
# ---------------------------
def show_pnl_page():
    st.title("📊 PNL Log")
    st.write("PNL Data will appear here once trades execute.")

# ---------------------------
# PAGE 3: CRYPTO BOT (GRID TRADING)
# ---------------------------
def show_crypto_bot_page():
    st.title("🤖 AI Grid Trading Bot")
    st_autorefresh(interval=30_000, key="grid_refresh") 
    
    usd_inr = st.session_state["usd_inr"]

    # --- A. TOP TABLE (Analysis) ---
    st.subheader("🔎 Live Market Analysis (USDT)")
    analysis_data = []
    for coin in CRYPTO_SYMBOLS_USD:
        hist = get_safe_crypto_data(coin, period="5d")
        if hist is not None:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((curr - prev) / prev) * 100
            rec = "HOLD"
            if change > 3.0: rec = "SELL (Overbought)"
            elif change < -3.0: rec = "BUY (Oversold)"
            volatility = (hist['High'] - hist['Low']).mean() / curr * 100
            analysis_data.append({
                "Coin": coin.replace("-USD", ""),
                "CMP (USDT)": f"${curr:,.2f}",
                "24h Change %": f"{change:+.2f}%",
                "Recommendation": rec,
                "Vol. Score": f"{volatility:.1f}/10"
            })
    
    if analysis_data:
        st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    else:
        st.warning("Fetching market data...")

    st.markdown("---")

    # --- B. GRID CONFIG PANEL (USDT) ---
    st.subheader("⚙️ Configure Grid Bot (USDT)")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        # SELECT BOX BACKGROUND IS WHITE VIA CSS ABOVE
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.2f}")
        st.caption(f"≈ ₹{curr_price * usd_inr:,.2f}")
        
        if st.button("🧠 Auto-Pick Best Settings"):
            if curr_price > 0:
                st.session_state['auto_lower'] = float(curr_price * 0.95)
                st.session_state['auto_upper'] = float(curr_price * 1.05)
                st.session_state['auto_grids'] = 5
                st.session_state['auto_tp'] = 2.0
                st.session_state['auto_sl'] = 3.0
                st.session_state['auto_inv'] = 100.0 # Default 100 USDT
                st.success("AI Settings Loaded!")
            else:
                st.error("Wait for price to load.")

    with c2:
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower Price (USDT)", value=st.session_state.get('auto_lower', 0.0))
        upper_p = col_b.number_input("Upper Price (USDT)", value=st.session_state.get('auto_upper', 0.0))
        
        col_c, col_d = st.columns(2)
        grids = col_c.number_input("No. of Grids", min_value=2, max_value=20, value=st.session_state.get('auto_grids', 5))
        invest = col_d.number_input("Investment (USDT)", value=st.session_state.get('auto_inv', 100.0))
        
        col_e, col_f = st.columns(2)
        tp_pct = col_e.number_input("Take Profit (%)", value=st.session_state.get('auto_tp', 2.0))
        sl_pct = col_f.number_input("Stop Loss (%)", value=st.session_state.get('auto_sl', 3.0))

    if st.button("▶️ Start Grid Bot"):
        if curr_price > 0 and lower_p < upper_p:
            bot_id = selected_coin
            entry_qty = invest / curr_price
            st.session_state["grid_bot_active"][bot_id] = {
                "coin": selected_coin,
                "entry_price": curr_price,
                "lower": lower_p,
                "upper": upper_p,
                "grids": grids,
                "qty": entry_qty,
                "invest": invest,
                "tp": tp_pct,
                "sl": sl_pct,
                "status": "Running",
                "start_time": dt.datetime.now().strftime("%H:%M:%S")
            }
            st.success(f"Bot Started for {selected_coin}!")
        else:
            st.error("Invalid Config or Price unavailable.")

    # --- C. ACTIVE BOT STATUS TABLE (USDT + INR) ---
    st.markdown("---")
    st.subheader("📍 Active Grid Bots (Dual Currency)")
    
    active_bots = st.session_state["grid_bot_active"]
    
    if active_bots:
        # Header (Widths adjusted for dual currency)
        h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([1.2, 1.2, 1.2, 2.0, 2.0, 2.0, 1.5, 1])
        h1.markdown("**Coin**")
        h2.markdown("**Entry**")
        h3.markdown("**CMP**")
        h4.markdown("**Inv. (USDT / INR)**")
        h5.markdown("**Value (USDT / INR)**")
        h6.markdown("**PnL (USDT / INR)**")
        h7.markdown("**Status**")
        h8.markdown("**Action**")
        
        st.markdown("<div style='border-bottom: 1px solid #ccc; margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        # Totals
        total_inv_usd = 0.0
        total_pnl_usd = 0.0

        for b_id, data in list(active_bots.items()):
            cp = get_current_price(data['coin'])
            
            if cp > 0:
                current_val_usd = data['qty'] * cp
                pnl_usd = current_val_usd - data['invest']
                pnl_pct = (pnl_usd / data['invest']) * 100
                
                # Update Totals
                total_inv_usd += data['invest']
                total_pnl_usd += pnl_usd

                status_text = "🟢 Running"
                if pnl_pct >= data['tp']: status_text = "✅ TP HIT"
                elif pnl_pct <= -data['sl']: status_text = "❌ SL HIT"
                
                # Conversion for display
                inv_inr = data['invest'] * usd_inr
                val_inr = current_val_usd * usd_inr
                pnl_inr = pnl_usd * usd_inr

                # Display Row
                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.2, 1.2, 1.2, 2.0, 2.0, 2
