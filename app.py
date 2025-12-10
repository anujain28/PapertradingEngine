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
        
        /* Custom Table Header for Bot List */
        .bot-header {
            font-weight: bold;
            border-bottom: 2px solid #000;
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        .bot-row {
            padding: 10px 0;
            border-bottom: 1px solid #eee;
            align-items: center;
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

if "grid_bot_active" not in st.session_state:
    st.session_state["grid_bot_active"] = {} 

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

    # --- A. TOP TABLE (Analysis) ---
    st.subheader("🔎 Live Market Analysis (30s Update)")
    analysis_data = []
    for coin in CRYPTO_SYMBOLS_USDT:
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

    # --- B. GRID CONFIG PANEL ---
    st.subheader("⚙️ Configure Grid Bot")
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USDT)
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.2f}")
        
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
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower Price", value=st.session_state.get('auto_lower', 0.0))
        upper_p = col_b.number_input("Upper Price", value=st.session_state.get('auto_upper', 0.0))
        
        col_c, col_d = st.columns(2)
        grids = col_c.number_input("No. of Grids", min_value=2, max_value=20, value=st.session_state.get('auto_grids', 5))
        invest = col_d.number_input("Investment (USDT)", value=st.session_state.get('auto_inv', 50.0))
        
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

    # --- C. ACTIVE BOT STATUS TABLE (CUSTOM LAYOUT) ---
    st.markdown("---")
    st.subheader("📍 Active Grid Bots")
    
    active_bots = st.session_state["grid_bot_active"]
    
    if active_bots:
        # 1. Header Row
        h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 1])
        h1.markdown("**Coin**")
        h2.markdown("**Entry**")
        h3.markdown("**CMP**")
        h4.markdown("**Inv.**")
        h5.markdown("**Pres. Val**")
        h6.markdown("**PnL**")
        h7.markdown("**Status**")
        h8.markdown("**Action**")
        
        st.markdown("<div style='border-bottom: 1px solid #ccc; margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        # Totals
        total_inv = 0.0
        total_curr = 0.0
        total_pnl = 0.0

        # 2. Data Rows
        # Iterate over a list of items to avoid runtime dict change errors
        for b_id, data in list(active_bots.items()):
            cp = get_current_price(data['coin'])
            
            # Calculations
            if cp > 0:
                current_val = data['qty'] * cp
                pnl = current_val - data['invest']
                pnl_pct = (pnl / data['invest']) * 100
                
                # Check TP/SL
                status_text = "🟢 Running"
                if pnl_pct >= data['tp']: status_text = "✅ TP HIT"
                elif pnl_pct <= -data['sl']: status_text = "❌ SL HIT"
                
                # Add to totals
                total_inv += data['invest']
                total_curr += current_val
                total_pnl += pnl
                
                # Render Row
                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 1])
                
                c1.write(data['coin'].replace("-USD",""))
                c2.write(f"${data['entry_price']:.2f}")
                c3.write(f"${cp:.2f}")
                c4.write(f"${data['invest']:.2f}")
                c5.write(f"${current_val:.2f}")
                
                # Color code PnL
                pnl_color = "green" if pnl >= 0 else "red"
                c6.markdown(f":{pnl_color}[${pnl:.2f}]")
                
                c7.write(status_text)
                
                # Individual Stop Button
                if c8.button("Stop 🟥", key=f"stop_{b_id}"):
                    del st.session_state["grid_bot_active"][b_id]
                    st.rerun()
        
        st.markdown("<div style='border-bottom: 1px solid #ccc; margin-top: 10px; margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        
        # 3. Summary Footer
        f1, f2, f3 = st.columns(3)
        f1.metric("Total Investment", f"${total_inv:,.2f}")
        f2.metric("Present Value", f"${total_curr:,.2f}")
        f3.metric("Total PnL", f"${total_pnl:,.2f}", delta_color="normal")
        
    else:
        st.info("No Active Grid Bots. Configure and start one above.")

# ---------------------------
# PAGE 4: CRYPTO DASHBOARD
# ---------------------------
def show_crypto_dashboard_page():
    st.title("🖥️ Global Crypto Dashboard")
    st_autorefresh(interval=300_000, key="dash_refresh")

    dash_coin = st.sidebar.selectbox("Select Asset", CRYPTO_SYMBOLS_USDT)
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
        
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
        
        fig.update_layout(
            height=500, margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor='black', paper_bgcolor='black',     
            xaxis=dict(showgrid=True, gridcolor='#444444', color='white'), 
            yaxis=dict(showgrid=True, gridcolor='#444444', color='white'), 
            font=dict(color='white')   
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data unavailable.")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Bitcoin_logo.svg/1200px-Bitcoin_logo.svg.png", width=50)
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Paper Trading", "PNL Log", "Crypto Bot", "Crypto Dashboard"])

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
        show_crypto_bot_page()
    elif page == "Crypto Dashboard":
        show_crypto_dashboard_page()

if __name__ == "__main__":
    main()
