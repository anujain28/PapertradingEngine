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
        /* 1. Main Page Background & Global Font Color */
        .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* 2. Sidebar Styling (Dark Grey/Black for Sidebar) */
        section[data-testid="stSidebar"] {
            background-color: #262730 !important; 
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] * {
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] div[role="radiogroup"] label {
            color: #ffffff !important;
        }

        /* 3. Force all text elements on Main Page to be black */
        p, h1, h2, h3, h4, h5, h6, span, div, label, li {
            color: #000000 !important;
        }

        /* 4. Metric Boxes - Light Grey Background */
        div[data-testid="metric-container"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            color: #000000 !important;
            border-radius: 8px;
        }

        /* 5. Tables - Light Grey Background & Black Text */
        div[data-testid="stDataFrame"] {
            background-color: #f0f2f6 !important;
            border: 1px solid #d1d5db;
            border-radius: 5px;
            padding: 5px;
        }

        /* 6. Buttons - Light Grey Background */
        .stButton > button {
            background-color: #e5e7eb !important;
            color: #000000 !important;
            border: 1px solid #9ca3af !important;
        }
        .stButton > button:hover {
            background-color: #d1d5db !important;
            color: #000000 !important;
            border-color: #6b7280 !important;
        }
        
        /* 7. Selectbox (The Box Itself) - Light Grey */
        div[data-baseweb="select"] > div {
            background-color: #f0f2f6 !important;
            color: #000000 !important;
            border: 1px solid #d1d5db;
        }

        /* 8. DROPDOWN MENU FIX (The Popup List) - White Background */
        div[data-baseweb="popover"] {
            background-color: #ffffff !important;
        }
        div[data-baseweb="menu"] {
            background-color: #ffffff !important;
        }
        div[role="option"], li[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        div[role="option"]:hover, li[role="option"]:hover {
            background-color: #f0f2f6 !important;
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
DB_PATH = "paper_trades.db"
CRYPTO_SYMBOLS_USDT = ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"]

# ---------------------------
# ROBUST DATA FETCHING
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

# ---------------------------
# STATE MANAGEMENT
# ---------------------------
if "state" not in st.session_state:
    st.session_state["state"] = {
        "capital": START_CAPITAL,
        "equity": START_CAPITAL,
        "positions": {},
    }

# Grid Bot Specific State
if "grid_bot_active" not in st.session_state:
    st.session_state["grid_bot_active"] = {} # Key: CoinSymbol, Value: Dict of params

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
    st_autorefresh(interval=30_000, key="grid_refresh") # 30s refresh

    # --- A. TOP TABLE (Analysis) ---
    st.subheader("🔎 Live Market Analysis (30s Update)")
    
    analysis_data = []
    for coin in CRYPTO_SYMBOLS_USDT:
        hist = get_safe_crypto_data(coin, period="5d")
        if hist is not None:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((curr - prev) / prev) * 100
            
            # Simple AI Logic for Recommendation
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

    # --- B. GRID CONFIG PANEL ---
    st.subheader("⚙️ Configure Grid Bot")
    
    c1, c2 = st.columns([1, 2])
    
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USDT)
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.2f}")
        
        # Brain Button Logic
        if st.button("🧠 Auto-Pick Best Settings"):
            if curr_price > 0:
                st.session_state['auto_lower'] = float(curr_price * 0.95)
                st.session_state['auto_upper'] = float(curr_price * 1.05)
                st.session_state['auto_grids'] = 5
                st.session_state['auto_tp'] = 2.0
                st.session_state['auto_sl'] = 3.0
                st.session_state['auto_inv'] = 50.0
                st.success("AI Settings Loaded!")
            else:
                st.error("Wait for price to load.")

    with c2:
        # Inputs (Use session state if auto-pick was clicked)
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower Price", value=st.session_state.get('auto_lower', 0.0))
        upper_p = col_b.number_input("Upper Price", value=st.session_state.get('auto_upper', 0.0))
        
        col_c, col_d = st.columns(2)
        grids = col_c.number_input("No. of Grids", min_value=2, max_value=20, value=st.session_state.get('auto_grids', 5))
        invest = col_d.number_input("Investment (USDT)", value=st.session_state.get('auto_inv', 50.0))
        
        col_e, col_f = st.columns(2)
        tp_pct = col_e.number_input("Take Profit (%)", value=st.session_state.get('auto_tp', 2.0))
        sl_pct = col_f.number_input("Stop Loss (%)", value=st.session_state.get('auto_sl', 3.0))

    # --- C. ACTION BUTTONS ---
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
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
                st.success(f"Grid Bot Started for {selected_coin}!")
            else:
                st.error("Invalid Config or Price unavailable.")

    with btn_col2:
        if st.button("⏹ Stop Grid Bot"):
            if selected_coin in st.session_state["grid_bot_active"]:
                del st.session_state["grid_bot_active"][selected_coin]
                st.warning(f"Bot stopped for {selected_coin}")

    # --- D. ACTIVE BOT STATUS TABLE ---
    st.subheader("📍 Active Grid Bots")
    
    active_bots = st.session_state["grid_bot_active"]
    if active_bots:
        bot_rows = []
        for b_id, data in active_bots.items():
            # Live PNL Calc
            cp = get_current_price(data['coin'])
            if cp > 0:
                pnl = (cp - data['entry_price']) * data['qty']
                pnl_pct = (pnl / data['invest']) * 100
                
                # Check TP/SL
                status = "Running"
                if pnl_pct >= data['tp']: status = "TAKE PROFIT HIT ✅"
                elif pnl_pct <= -data['sl']: status = "STOP LOSS HIT ❌"
                
                bot_rows.append({
                    "Coin": data['coin'],
                    "CMP": f"${cp:,.2f}",
                    "Entry": f"${data['entry_price']:,.2f}",
                    "Lower": data['lower'],
                    "Upper": data['upper'],
                    "Qty": f"{data['qty']:.4f}",
                    "Inv.": f"${data['invest']}",
                    "TP/SL": f"{data['tp']}% / {data['sl']}%",
                    "PnL (USDT)": f"${pnl:.2f}",
                    "Status": status,
                    "Started": data['start_time']
                })
        
        st.dataframe(pd.DataFrame(bot_rows), use_container_width=True)
    else:
        st.info("No Active Grid Bots.")


# ---------------------------
# PAGE 4: CRYPTO DASHBOARD
# ---------------------------
def show_crypto_dashboard_page():
    st.title("🖥️ Global Crypto Dashboard")
    st_autorefresh(interval=300_000, key="dash_refresh")

    # Sidebar Selection specific to this page
    dash_coin = st.sidebar.selectbox("Select Asset", ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "DOGE-USD", "ADA-USD"])
    time_range = st.sidebar.select_slider("Time Range", options=["1mo", "3mo", "6mo", "1y", "5y", "max"])

    data = get_safe_crypto_data(dash_coin, period=time_range)
    
    if data is not None:
        curr = data['Close'].iloc[-1]
        prev = data['Close'].iloc[-2]
        chg = ((curr - prev)/prev)*100
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Current Price", f"${curr:,.2f}", f"{chg:.2f}%")
        m2.metric("24h High", f"${data['High'].iloc[-1]:,.2f}")
        m3.metric("Volume", f"{data['Volume'].iloc[-1]:,.0f}")
        
        st.subheader(f"Price Chart: {dash_coin}")
        
        # Black Chart
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
        
        fig.update_layout(
            height=500, 
            margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor='black',      
            paper_bgcolor='black',     
            xaxis=dict(showgrid=True, gridcolor='#444444', color='white'), 
            yaxis=dict(showgrid=True, gridcolor='#444444', color='white'), 
            font=dict(color='white')   
        )
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📄 View Raw Data"):
            st.dataframe(data.sort_index(ascending=False).head(50), use_container_width=True)

    else:
        st.error("Data unavailable. Try a different coin or timeframe.")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Bitcoin_logo.svg/1200px-Bitcoin_logo.svg.png", width=50)
    st.sidebar.title("Navigation")
    
    page = st.sidebar.radio("Go to", ["Paper Trading", "PNL Log", "Crypto Bot", "Crypto Dashboard"])

    # Threads initialization (Safety Check)
    if not st.session_state.get("loop_started", False):
        st.session_state["loop_started"] = True
    
    # Crypto Loop
    if CRYPTO_BOT_AVAILABLE and not st.session_state.get("crypto_loop_started", False):
        t_crypto = threading.Thread(target=crypto_trading_loop, daemon=True)
        t_crypto.start()
        st.session_state["crypto_loop_started"] = True

    # Page Routing
    if page == "Paper Trading":
        show_paper_trading_page()
    elif page == "PNL Log":
        show_pnl_page()
    elif page == "Crypto Bot":
        show_crypto_bot_page()
    elif page == "Crypto Dashboard":
        show_crypto_dashboard_page()

if __name__ == "__main__":
    main()
