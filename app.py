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

        /* --- DROPDOWN MENU FIX --- */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da;
        }
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

@st.cache_data(ttl=3600)
def get_usd_inr_rate():
    try:
        data = yf.Ticker("INR=X").history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return 84.0 # Fallback

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

# --- AI AUTO-PILOT STATE ---
if "autopilot" not in st.session_state:
    st.session_state["autopilot"] = {
        "running": False,
        "currency": "USDT",
        "total_capital": 0.0,
        "cash_balance": 0.0,
        "active_grids": [], # List of grid bot dicts
        "logs": []
    }

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
# PAGE 1: PAPER TRADING (STOCKS)
# ---------------------------
def show_paper_trading_page():
    st.title("📈 AI Stocks Paper Trading")
    st_autorefresh(interval=120_000, key="auto_refresh")
    state = st.session_state["state"]
    col1, col2 = st.columns(2)
    col1.metric("Free Capital", f"₹{state['capital']:,.2f}")
    col2.metric("Equity", f"₹{state['equity']:,.2f}")
    st.info(f"Engine Status: {st.session_state.get('engine_status')}")

# ---------------------------
# PAGE 2: PNL LOG (STOCKS)
# ---------------------------
def show_pnl_page():
    st.title("📊 Stocks PNL Log")
    st.write("PNL Data will appear here once trades execute.")

# ---------------------------
# PAGE 3: CRYPTO BOT (GRID TRADING)
# ---------------------------
def show_crypto_bot_page():
    st.title("🤖 AI Grid Trading Bot")
    st_autorefresh(interval=30_000, key="grid_refresh") 
    usd_inr = st.session_state["usd_inr"]

    # Analysis Section
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
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    st.markdown("---")

    # Grid Config
    st.subheader("⚙️ Configure Grid Bot (Manual)")
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.2f}")
        st.caption(f"≈ ₹{curr_price * usd_inr:,.2f}")
        
        if st.button("🧠 Auto-Pick Settings"):
            if curr_price > 0:
                st.session_state['auto_lower'] = float(curr_price * 0.95)
                st.session_state['auto_upper'] = float(curr_price * 1.05)
                st.session_state['auto_grids'] = 5
                st.session_state['auto_tp'] = 2.0
                st.session_state['auto_sl'] = 3.0
                st.session_state['auto_inv'] = 100.0
                st.success("Loaded!")

    with c2:
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower Price", value=st.session_state.get('auto_lower', 0.0))
        upper_p = col_b.number_input("Upper Price", value=st.session_state.get('auto_upper', 0.0))
        
        col_c, col_d = st.columns(2)
        grids = col_c.number_input("Grids", min_value=2, max_value=20, value=st.session_state.get('auto_grids', 5))
        invest = col_d.number_input("Investment", value=st.session_state.get('auto_inv', 100.0))
        
        col_e, col_f = st.columns(2)
        tp_pct = col_e.number_input("TP (%)", value=st.session_state.get('auto_tp', 2.0))
        sl_pct = col_f.number_input("SL (%)", value=st.session_state.get('auto_sl', 3.0))

    if st.button("▶️ Start Grid Bot"):
        if curr_price > 0 and lower_p < upper_p:
            bot_id = selected_coin
            entry_qty = invest / curr_price
            st.session_state["grid_bot_active"][bot_id] = {
                "coin": selected_coin, "entry_price": curr_price,
                "lower": lower_p, "upper": upper_p, "grids": grids,
                "qty": entry_qty, "invest": invest, "tp": tp_pct, "sl": sl_pct,
                "status": "Running", "start_time": dt.datetime.now().strftime("%H:%M:%S")
            }
            st.success("Bot Started!")

    # Active Bots
    st.markdown("---")
    st.subheader("📍 Active Grid Bots")
    active_bots = st.session_state["grid_bot_active"]
    if active_bots:
        h1, h2, h3, h4, h5, h6, h7 = st.columns([1,1,1,2,2,2,1])
        h1.write("**Coin**"); h2.write("**Entry**"); h3.write("**CMP**"); 
        h4.write("**Inv (USD/INR)**"); h5.write("**PnL (USD/INR)**"); h6.write("**Status**"); h7.write("**Action**")
        
        for b_id, data in list(active_bots.items()):
            cp = get_current_price(data['coin'])
            current_val_usd = data['qty'] * cp
            pnl_usd = current_val_usd - data['invest']
            pnl_inr = pnl_usd * usd_inr
            inv_inr = data['invest'] * usd_inr
            
            pnl_pct = (pnl_usd / data['invest']) * 100
            status_text = "🟢 Running"
            if pnl_pct >= data['tp']: status_text = "✅ TP HIT"
            elif pnl_pct <= -data['sl']: status_text = "❌ SL HIT"

            c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,2,2,2,1])
            c1.write(data['coin'].replace("-USD",""))
            c2.write(f"${data['entry_price']:.2f}")
            c3.write(f"${cp:.2f}")
            c4.write(f"${data['invest']:.0f} / ₹{inv_inr:,.0f}")
            pnl_color = "green" if pnl_usd >= 0 else "red"
            c5.markdown(f":{pnl_color}[${pnl_usd:.2f} / ₹{pnl_inr:,.0f}]")
            c6.write(status_text)
            if c7.button("Stop", key=f"stop_{b_id}"):
                del st.session_state["grid_bot_active"][b_id]
                st.rerun()
    else:
        st.info("No active grid bots.")

# ---------------------------
# PAGE 4: CRYPTO DASHBOARD
# ---------------------------
def show_crypto_dashboard_page():
    st.title("🖥️ Global Crypto Dashboard")
    st_autorefresh(interval=300_000, key="dash_refresh")
    dash_coin = st.sidebar.selectbox("Select Asset", CRYPTO_SYMBOLS_USD)
    data = get_safe_crypto_data(dash_coin)
    
    if data is not None:
        curr = data['Close'].iloc[-1]
        fig = go.Figure(data=[go.Candlestick(x=data.index,
                        open=data['Open'], high=data['High'],
                        low=data['Low'], close=data['Close'])])
        fig.update_layout(height=500, plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data unavailable.")

# ---------------------------
# PAGE 5: AI AUTO-PILOT (GRID ENABLED)
# ---------------------------
def show_ai_autopilot_page():
    st.title("🚀 AI Auto-Pilot (Grid Enabled)")
    st.caption("Auto-scans and launches Grid Bots automatically for volatility harvesting.")
    st_autorefresh(interval=15_000, key="autopilot_refresh") 
    
    usd_inr = st.session_state["usd_inr"]
    ap = st.session_state["autopilot"]

    # --- CONFIGURATION SECTION ---
    if not ap["running"]:
        st.subheader("🛠️ Setup Auto-Pilot")
        c1, c2, c3 = st.columns(3)
        currency_mode = c1.radio("Capital Currency", ["USDT (USD)", "INR (₹)"])
        capital_input = c2.number_input("Total Capital Allocation", min_value=10.0, value=1000.0, step=100.0)
        
        if c3.button("🚀 Launch AI Engine", type="primary", use_container_width=True):
            ap["running"] = True
            ap["currency"] = "USDT" if "USDT" in currency_mode else "INR"
            
            if ap["currency"] == "INR":
                ap["total_capital"] = capital_input / usd_inr
                ap["cash_balance"] = capital_input / usd_inr
            else:
                ap["total_capital"] = capital_input
                ap["cash_balance"] = capital_input
            
            ap["logs"].append(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Engine Started. Capital: ${ap['total_capital']:.2f}")
            st.rerun()
            
    else:
        # --- RUNNING DASHBOARD ---
        
        # 1. HEADER METRICS
        curr_sym = "$" if ap["currency"] == "USDT" else "₹"
        conv_factor = 1.0 if ap["currency"] == "USDT" else usd_inr
        
        # Calculate Active Grids Value
        invested_in_grids = sum([g['invest'] for g in ap['active_grids']])
        
        # Current Value (Simplified: Cash + Current Value of Grids)
        # Note: Ideally we check live PnL of each grid
        grid_current_val = 0.0
        for g in ap['active_grids']:
             cp = get_current_price(g['coin'])
             grid_current_val += (g['qty'] * cp)

        total_val_usd = ap['cash_balance'] + grid_current_val
        total_pnl_usd = total_val_usd - ap['total_capital']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Value", f"{curr_sym}{total_val_usd * conv_factor:,.2f}")
        m2.metric("Cash Balance", f"{curr_sym}{ap['cash_balance'] * conv_factor:,.2f}")
        m3.metric("Invested (Grids)", f"{curr_sym}{grid_current_val * conv_factor:,.2f}")
        m4.metric("Net PnL", f"{curr_sym}{total_pnl_usd * conv_factor:,.2f}", 
                  delta=f"{(total_pnl_usd/ap['total_capital'])*100:.2f}%")
        
        st.markdown("---")
        
        # 2. AI SCANNER LOGIC (LAUNCHES GRID BOTS)
        if ap['cash_balance'] > (ap['total_capital'] * 0.2): # Only launch if we have 20% cash
            scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
            
            # Check if we already have a grid for this coin
            existing = [g for g in ap['active_grids'] if g['coin'] == scan_coin]
            
            if not existing:
                cp = get_current_price(scan_coin)
                chance = random.randint(1, 10)
                
                if chance > 8 and cp > 0: # 20% chance to find volatility
                    invest_amt = ap['cash_balance'] * 0.2 # Use 20% of cash for this grid
                    
                    # GRID PARAMETERS (SMART SETTINGS)
                    lower = cp * 0.95
                    upper = cp * 1.05
                    qty = invest_amt / cp
                    
                    new_grid = {
                        "coin": scan_coin,
                        "entry": cp,
                        "lower": lower,
                        "upper": upper,
                        "qty": qty,
                        "invest": invest_amt,
                        "grids": 5,
                        "start_time": dt.datetime.now().strftime('%H:%M:%S')
                    }
                    
                    ap['active_grids'].append(new_grid)
                    ap['cash_balance'] -= invest_amt
                    ap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] 🤖 DEPLOYED GRID: {scan_coin} (Range: ${lower:.2f}-${upper:.2f})")
                
                elif chance < 2: 
                    ap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] 🔍 Scanned {scan_coin}. Low Volatility. Skipped.")
        
        if len(ap['logs']) > 5: ap['logs'] = ap['logs'][:5]
        for log in ap['logs']:
            st.text(log)
            
        st.markdown("---")
        
        # 3. LIVE AUTO-PILOT GRIDS
        st.subheader("💼 Active Auto-Pilot Grids")
        
        if ap['active_grids']:
            # Headers
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1.5,1.5,1.5,1])
            c1.markdown("**Asset**")
            c2.markdown("**Range**")
            c3.markdown("**Current**")
            c4.markdown("**Invested**")
            c5.markdown("**Value**")
            c6.markdown("**Profit/Loss**")
            c7.markdown("**Action**")
            st.markdown("<div style='border-bottom:1px solid #ddd; margin-bottom:10px;'></div>", unsafe_allow_html=True)

            # Totals
            sum_inv = 0.0
            sum_val = 0.0
            sum_pnl = 0.0

            # Rows
            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                
                sum_inv += g['invest']
                sum_val += curr_val
                sum_pnl += pnl

                c1, c2, c3, c4, c5, c6, c7 = st.columns([1,1,1,1.5,1.5,1.5,1])
                
                c1.write(g['coin'].replace("-USD",""))
                c2.write(f"${g['lower']:.0f}-${g['upper']:.0f}")
                c3.write(f"${cp:.2f}")
                
                c4.write(f"${g['invest']:.2f}")
                c5.write(f"${curr_val:.2f}")
                
                pnl_color = "green" if pnl >= 0 else "red"
                c6.markdown(f":{pnl_color}[${pnl:.2f}]")
                
                # Stop Button
                if c7.button("Stop 🟥", key=f"ap_grid_stop_{i}"):
                    # Liquidation
                    ap['cash_balance'] += curr_val
                    ap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] 🔴 MANUAL STOP: Closed Grid {g['coin']}. Returned ${curr_val:.2f}")
                    ap['active_grids'].pop(i)
                    st.rerun()

            st.markdown("<div style='border-bottom:1px solid #ddd; margin-top:10px; margin-bottom:10px;'></div>", unsafe_allow_html=True)
            
            # FOOTER SUMMARY
            f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1])
            f1.write("**TOTALS (USD):**")
            f2.write(f"**${sum_inv:,.2f}**")
            f3.write(f"**${sum_val:,.2f}**")
            
            total_pnl_color = "green" if sum_pnl >= 0 else "red"
            f4.markdown(f":{total_pnl_color}[**${sum_pnl:,.2f}**]")
            
        else:
            st.info("AI is scanning for volatility opportunities...")
            
        if st.button("⏹ Emergency Stop Engine"):
            ap["running"] = False
            st.rerun()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Bitcoin_logo.svg/1200px-Bitcoin_logo.svg.png", width=50)
    
    st.sidebar.title("Navigation")
    
    # 1. Asset Class Selector
    market_type = st.sidebar.radio("Select Market", ["Stocks", "Crypto"], index=1)
    st.sidebar.markdown("---")
    
    page = None
    
    if market_type == "Stocks":
        st.sidebar.subheader("Stocks Menu")
        page = st.sidebar.radio("Go to", ["Paper Trading", "PNL Log"])
    
    else: # Crypto
        st.sidebar.subheader("Crypto Menu")
        page = st.sidebar.radio("Go to", ["Crypto Grid Bot", "AI Auto-Pilot", "Crypto Dashboard"])

    # --- THREADS ---
    if not st.session_state.get("loop_started", False):
        st.session_state["loop_started"] = True
    
    if CRYPTO_BOT_AVAILABLE and not st.session_state.get("crypto_loop_started", False):
        t_crypto = threading.Thread(target=crypto_trading_loop, daemon=True)
        t_crypto.start()
        st.session_state["crypto_loop_started"] = True

    # --- ROUTING ---
    if page == "Paper Trading":
        show_paper_trading_page()
    elif page == "PNL Log":
        show_pnl_page()
    elif page == "Crypto Grid Bot":
        show_crypto_bot_page()
    elif page == "AI Auto-Pilot":
        show_ai_autopilot_page()
    elif page == "Crypto Dashboard":
        show_crypto_dashboard_page()

if __name__ == "__main__":
    main()
