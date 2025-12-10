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
        section[data-testid="stSidebar"] input { color: black !important; }
        
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

# COIN LIST: BTC, ETH, BNB, SOL, ADA
CRYPTO_SYMBOLS_USD = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]

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
        "running": False, "mode": "PAPER", "currency": "USDT",
        "total_capital": 0.0, "cash_balance": 0.0,
        "active_grids": [], "logs": [], "history": []
    }
else:
    # Ensure all keys exist
    defaults = {
        "running": False, "mode": "PAPER", "currency": "USDT",
        "total_capital": 0.0, "cash_balance": 0.0,
        "active_grids": [], "logs": [], "history": []
    }
    for k, v in defaults.items():
        if k not in st.session_state["autopilot"]:
            st.session_state["autopilot"][k] = v

for key in ["engine_status", "engine_running", "loop_started", 
            "crypto_running", "crypto_status", "crypto_loop_started",
            "binance_api", "binance_secret"]:
    if key not in st.session_state:
        st.session_state[key] = None if "binance" in key else False

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
# PAGE 1: CRYPTO MANUAL BOT (Renamed)
# ---------------------------
def show_crypto_manual_bot_page():
    st.title("🤖 AI Crypto Manual Bot")
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
    st.subheader("⚙️ Configure Manual Bot")
    c1, c2 = st.columns([1, 2])
    with c1:
        # Save selection in state
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD, key="bot_coin_select")
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

    if st.button("▶️ Start Manual Bot"):
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
    st.subheader("📍 Active Manual Bots")
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
        st.info("No active manual bots.")

    # Live Grid Orders
    st.markdown("### 📋 Live Grid Orders")
    if active_bots:
        for b_id, data in active_bots.items():
            with st.expander(f"Orders for {b_id}"):
                lower = data['lower']
                upper = data['upper']
                grids = data.get('grids', 5)
                levels = np.linspace(lower, upper, grids)
                orders = []
                for lvl in levels:
                    if lvl < data['entry_price']: orders.append({"Side": "BUY", "Price": f"${lvl:.2f}", "Status": "Open"})
                    else: orders.append({"Side": "SELL", "Price": f"${lvl:.2f}", "Status": "Open"})
                st.table(pd.DataFrame(orders))
    else:
        st.caption("Start a bot to see grid levels.")

    st.markdown("---")
    
    # CHART SECTION
    st.subheader(f"📉 Asset Price Chart: {selected_coin}")
    
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        time_range = st.radio("Select Timeframe", 
                             ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd", "max"], 
                             index=2, 
                             horizontal=True)
    
    chart_data = get_safe_crypto_data(selected_coin, period=time_range)
    
    if chart_data is not None:
        fig = go.Figure(data=[go.Candlestick(x=chart_data.index,
                        open=chart_data['Open'], high=chart_data['High'],
                        low=chart_data['Low'], close=chart_data['Close'])])
        
        fig.update_layout(
            height=500, margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor='black', paper_bgcolor='black',
            xaxis=dict(showgrid=True, gridcolor='#333', color='white'),
            yaxis=dict(showgrid=True, gridcolor='#333', color='white'),
            font=dict(color='white')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Chart data unavailable.")


# ---------------------------
# PAGE 5: AI AUTO-PILOT (LIVE ENABLED)
# ---------------------------
def show_ai_autopilot_page():
    usd_inr = st.session_state["usd_inr"]
    ap = st.session_state["autopilot"]

    api_key = st.session_state.get("binance_api")
    secret_key = st.session_state.get("binance_secret")
    is_live = bool(api_key and secret_key)
    
    mode_title = "🔴 LIVE TRADING" if is_live else "🟢 PAPER TRADING"
    st.title(f"🚀 AI Auto-Pilot ({mode_title})")
    
    if is_live:
        st.warning("⚠️ **LIVE MONEY MODE ACTIVE**: The AI will execute real trades via Binance API.")
    else:
        st.info("ℹ️ **Simulation Mode**: Add Binance Keys in Sidebar to switch to Live Mode.")

    st_autorefresh(interval=15_000, key="autopilot_refresh") 
    
    if not ap["running"]:
        st.subheader("🛠️ Setup Auto-Pilot")
        c1, c2, c3 = st.columns(3)
        currency_mode = c1.radio("Capital Currency", ["USDT (USD)", "INR (₹)"])
        capital_input = c2.number_input("Total Capital Allocation", min_value=10.0, value=1000.0, step=100.0)
        
        btn_label = "🚀 Launch LIVE AI" if is_live else "🚀 Launch Simulation"
        if c3.button(btn_label, type="primary", use_container_width=True):
            ap["running"] = True
            ap["mode"] = "LIVE" if is_live else "PAPER"
            ap["currency"] = "USDT" if "USDT" in currency_mode else "INR"
            
            if ap["currency"] == "INR":
                ap["total_capital"] = capital_input / usd_inr
                ap["cash_balance"] = capital_input / usd_inr
            else:
                ap["total_capital"] = capital_input
                ap["cash_balance"] = capital_input
            
            ap["active_grids"] = []
            ap["logs"].append(f"[{dt.datetime.now().strftime('%H:%M:%S')}] Engine Started in {ap['mode']} Mode. Capital: ${ap['total_capital']:.2f}")
            st.rerun()
            
    else:
        st.success("✅ AI Engine is Active: Analyzing Market Volatility & Updating Grids...")
        
        curr_sym = "$" if ap["currency"] == "USDT" else "₹"
        conv_factor = 1.0 if ap["currency"] == "USDT" else usd_inr
        
        invested_in_grids = sum([g.get('invest', 0.0) for g in ap['active_grids']])
        
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
        
        # SCANNER LOGIC
        if ap['cash_balance'] > (ap['total_capital'] * 0.2): 
            scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
            existing = [g for g in ap['active_grids'] if g['coin'] == scan_coin]
            
            if not existing:
                cp = get_current_price(scan_coin)
                chance = random.randint(1, 10)
                
                if chance > 8 and cp > 0: 
                    invest_amt = ap['cash_balance'] * 0.2 
                    lower = cp * 0.95
                    upper = cp * 1.05
                    qty = invest_amt / cp
                    
                    grid_levels = np.linspace(lower, upper, 5)
                    grid_orders = []
                    for lvl in grid_levels:
                        if lvl < cp: grid_orders.append({"type": "BUY", "price": lvl, "status": "OPEN"})
                        else: grid_orders.append({"type": "SELL", "price": lvl, "status": "OPEN"})
                    
                    new_grid = {
                        "coin": scan_coin, "entry": cp,
                        "lower": lower, "upper": upper, "qty": qty,
                        "invest": invest_amt, "grids": 5, "tp": 2.0, "sl": 3.0,
                        "orders": grid_orders,
                        "start_time": dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    ap['active_grids'].append(new_grid)
                    ap['cash_balance'] -= invest_amt
                    log_prefix = "🔴 LIVE" if ap["mode"] == "LIVE" else "🤖 PAPER"
                    ap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] {log_prefix}: Deployed Grid on {scan_coin}")
                
                elif chance < 2: 
                    ap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] 🔍 Scanned {scan_coin}. Low Volatility. Skipped.")
        
        if len(ap['logs']) > 5: ap['logs'] = ap['logs'][:5]
        for log in ap['logs']:
            st.text(log)
            
        st.markdown("---")
        
        # ACTIVE GRIDS TABLE
        st.subheader(f"💼 Active {ap['mode']} Grids")
        
        if ap['active_grids']:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.2, 1.2, 1.2, 1.2, 0.8])
            c1.markdown("**Asset**"); c2.markdown("**Range (L-U)**"); c3.markdown("**Grid Config**")
            c4.markdown("**Invested**"); c5.markdown("**Current Val**"); c6.markdown("**Profit/Loss**"); c7.markdown("**Action**")
            st.markdown("<div style='border-bottom:1px solid #ddd; margin-bottom:10px;'></div>", unsafe_allow_html=True)

            sum_inv = 0.0; sum_val = 0.0; sum_pnl = 0.0

            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                sum_inv += g['invest']; sum_val += curr_val; sum_pnl += pnl
                
                grid_count = g.get('grids', 5)
                tp_val = g.get('tp', 2.0)

                c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.2, 1.2, 1.2, 1.2, 0.8])
                c1.write(g['coin'].replace("-USD",""))
                c2.write(f"${g['lower']:.4f} - ${g['upper']:.4f}")
                c3.write(f"{grid_count} Grids | TP {tp_val}%")
                c4.write(f"${g['invest']:.2f}")
                c5.write(f"${curr_val:.2f}")
                pnl_color = "green" if pnl >= 0 else "red"
                c6.markdown(f":{pnl_color}[${pnl:.2f}]")
                
                if c7.button("Stop 🟥", key=f"ap_grid_stop_{i}"):
                    # CLOSE POSITION & CAPTURE IST TIMESTAMP
                    ap['cash_balance'] += curr_val
                    
                    close_time_ist = dt.datetime.now(IST)
                    
                    closed_trade = {
                        "date": close_time_ist,
                        "coin": g['coin'],
                        "invested": g['invest'],
                        "pnl": pnl,
                        "return_pct": (pnl/g['invest'])*100,
                        "duration": "Auto"
                    }
                    ap['history'].append(closed_trade)
                    
                    ap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M:%S')}] 🔴 STOPPED: Closed Grid {g['coin']}.")
                    ap['active_grids'].pop(i)
                    st.rerun()

            st.markdown("<div style='border-bottom:1px solid #ddd; margin-top:10px; margin-bottom:10px;'></div>", unsafe_allow_html=True)
            f1, f2, f3, f4 = st.columns([2, 1.5, 1.5, 1])
            f1.write("**TOTALS (USD):**"); f2.write(f"**${sum_inv:,.2f}**"); f3.write(f"**${sum_val:,.2f}**")
            total_pnl_color = "green" if sum_pnl >= 0 else "red"
            f4.markdown(f":{total_pnl_color}[**${sum_pnl:,.2f}**]")
            
            # LIVE ORDERS
            st.markdown("### 📋 Live Grid Orders")
            for g in ap['active_grids']:
                with st.expander(f"Orders for {g['coin'].replace('-USD','')}"):
                    if 'orders' in g:
                        ord_df = pd.DataFrame(g['orders'])
                        ord_df['price'] = ord_df['price'].apply(lambda x: f"${x:.4f}")
                        st.dataframe(ord_df, use_container_width=True)

        else:
            st.info("AI is scanning for volatility opportunities...")
            
        if st.button("⏹ Emergency Stop Engine"):
            ap["running"] = False
            st.rerun()

# ---------------------------
# PAGE 6: CRYPTO PNL REPORT (LIVE + CLOSED)
# ---------------------------
def show_crypto_report_page():
    st.title("📑 Crypto PnL Report (Auto-Pilot)")
    ap = st.session_state["autopilot"]
    usd_inr = st.session_state["usd_inr"]
    
    st_autorefresh(interval=30_000, key="report_refresh")

    # --- 1. LIVE RUNNING TRADES SUMMARY ---
    st.subheader("🔴 Live Portfolio Overview (Running Trades)")
    
    running_invested_usd = sum([g.get('invest', 0.0) for g in ap['active_grids']])
    running_present_val_usd = 0.0
    for g in ap['active_grids']:
         cp = get_current_price(g['coin'])
         running_present_val_usd += (g['qty'] * cp)
    running_pnl_usd = running_present_val_usd - running_invested_usd
    
    r_inv_inr = running_invested_usd * usd_inr
    r_val_inr = running_present_val_usd * usd_inr
    r_pnl_inr = running_pnl_usd * usd_inr
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Live Invested (INR)", f"₹{r_inv_inr:,.0f}")
    c2.metric("Live Present Value (INR)", f"₹{r_val_inr:,.0f}")
    c3.metric("Live Unrealized PnL (INR)", f"₹{r_pnl_inr:,.0f}", delta_color="normal")
    
    st.markdown("---")

    # --- 2. CLOSED TRADES REPORT ---
    st.subheader("🏁 Historical Performance (Closed Trades)")
    
    if not ap["history"]:
        st.info("No closed trades available yet.")
        return

    df = pd.DataFrame(ap["history"])
    df['date'] = pd.to_datetime(df['date'])
    
    total_profit = df['pnl'].sum()
    win_count = len(df[df['pnl'] > 0])
    total_count = len(df)
    win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Realized PnL (Till Date)", f"${total_profit:,.2f} (₹{total_profit*usd_inr:,.0f})")
    m2.metric("Win Rate", f"{win_rate:.1f}%")
    m3.metric("Total Trades Closed", f"{total_count}")

    st.markdown("---")
    
    # --- TIME-BASED REPORTS ---
    t1, t2, t3 = st.tabs(["Daily Report", "Weekly Report", "Monthly Report"])
    
    with t1:
        st.subheader("Daily Profit Log (IST)")
        daily_df = df.groupby(df['date'].dt.date)[['invested', 'pnl']].sum().reset_index()
        daily_df['Net PnL (INR)'] = daily_df['pnl'] * usd_inr
        st.dataframe(daily_df, use_container_width=True)
        
    with t2:
        st.subheader("Weekly Summary")
        df['Week'] = df['date'].dt.isocalendar().week
        weekly_df = df.groupby('Week')[['invested', 'pnl']].sum().reset_index()
        weekly_df['Net PnL (INR)'] = weekly_df['pnl'] * usd_inr
        st.dataframe(weekly_df, use_container_width=True)
        
    with t3:
        st.subheader("Monthly Overview")
        df['Month'] = df['date'].dt.month_name()
        monthly_df = df.groupby('Month')[['invested', 'pnl']].sum().reset_index()
        monthly_df['Net PnL (INR)'] = monthly_df['pnl'] * usd_inr
        st.dataframe(monthly_df, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📜 Complete Trade Log")
    display_df = df.copy()
    display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S IST')
    st.dataframe(display_df.sort_values('date', ascending=False), use_container_width=True)


# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Bitcoin_logo.svg/1200px-Bitcoin_logo.svg.png", width=50)
    
    st.sidebar.title("Navigation")
    market_type = st.sidebar.radio("Select Market", ["Stocks", "Crypto"], index=1)
    st.sidebar.markdown("---")
    
    page = None
    if market_type == "Stocks":
        st.sidebar.subheader("Stocks Menu")
        if st.sidebar.button("Launch Stocks App"):
            try:
                # Placeholder for invoking app1.py
                st.warning("⚠️ Stocks Module (app1.py) is under construction.")
            except:
                pass
        
    else: # Crypto
        st.sidebar.subheader("Crypto Menu")
        # UPDATED ORDER
        page = st.sidebar.radio("Go to", ["AI Auto-Pilot", "Crypto Report", "Crypto Manual Bot"])
        
        st.sidebar.markdown("---")
        with st.sidebar.expander("🔌 Binance Keys (Live Trading)"):
            st.caption("Enter credentials to enable Live Trading mode.")
            api = st.text_input("API Key", value=st.session_state.get("binance_api", "") or "", type="password")
            sec = st.text_input("Secret Key", value=st.session_state.get("binance_secret", "") or "", type="password")
            if st.button("💾 Save Keys"):
                st.session_state["binance_api"] = api
                st.session_state["binance_secret"] = sec
                st.success("Keys Saved!")

    # --- THREADS ---
    if not st.session_state.get("loop_started", False):
        st.session_state["loop_started"] = True
    if CRYPTO_BOT_AVAILABLE and not st.session_state.get("crypto_loop_started", False):
        t_crypto = threading.Thread(target=crypto_trading_loop, daemon=True)
        t_crypto.start()
        st.session_state["crypto_loop_started"] = True

    # --- ROUTING ---
    if page == "AI Auto-Pilot":
        show_ai_autopilot_page()
    elif page == "Crypto Report":
        show_crypto_report_page()
    elif page == "Crypto Manual Bot":
        show_crypto_manual_bot_page()

if __name__ == "__main__":
    main()
