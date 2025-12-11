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

# --- IMPORT CRYPTO BOT ---
try:
    from crypto_bot import (
        init_crypto_state, crypto_trading_loop,
        get_crypto_positions
    )
    CRYPTO_BOT_AVAILABLE = True
except ImportError:
    CRYPTO_BOT_AVAILABLE = False

# ---------------------------
# PAGE CONFIG + GLOBAL STYLE
# ---------------------------
st.set_page_config(page_title="AI Crypto Trading", layout="wide", page_icon="📈")

def apply_custom_style():
    st.markdown("""
        <style>
        /* Global Styles */
        .stApp { background-color: #ffffff !important; color: #000000 !important; }
        p, h1, h2, h3, h4, h5, h6, span, div, label, li { color: #000000 !important; }
        
        /* Sidebar Container */
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        
        /* --- SIDEBAR INPUTS (High Contrast) --- */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            caret-color: #ffffff !important;
            border: 1px solid #666 !important;
        }
        section[data-testid="stSidebar"] label {
            color: #ffffff !important;
        }
        
        /* --- SIDEBAR EXPANDER FIX (Black BG, White Text) --- */
        section[data-testid="stSidebar"] div[data-testid="stExpander"] details summary {
            background-color: #333 !important; /* Dark Grey Header */
            color: #ffffff !important;
            border: 1px solid #555;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] div[role="group"] {
            background-color: #262730 !important; /* Match Sidebar BG */
            color: #ffffff !important;
        }
        section[data-testid="stSidebar"] div[data-testid="stExpander"] p,
        section[data-testid="stSidebar"] div[data-testid="stExpander"] span {
            color: #ffffff !important;
        }

        /* --- MAIN PAGE EXPANDER (White BG, Black Text for Grid Orders) --- */
        /* Only targets expanders NOT in the sidebar */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f8f9fa !important;
            color: #000000 !important;
            border: 1px solid #dee2e6;
        }
        .main div[data-testid="stExpander"] div[role="group"] {
            background-color: #ffffff !important;
        }
        .main div[data-testid="stExpander"] table, 
        .main div[data-testid="stExpander"] td, 
        .main div[data-testid="stExpander"] th,
        .main div[data-testid="stExpander"] div {
            color: #000000 !important;
        }

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
        
        /* Dataframes */
        div[data-testid="stDataFrame"] {
            background-color: #ffffff !important;
        }
        div[data-testid="stDataFrame"] div[class*="stDataFrame"] {
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
CRYPTO_SYMBOLS_USD = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]

# ---------------------------
# DATA FETCHING & AI ANALYSIS
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
    return 84.0 

# --- AI ALGORITHM: Technical Analysis ---
def calculate_ai_score(symbol):
    """
    Analyzes chart data to generate a trading score (-10 to +10).
    Factors: RSI, SMA Trend, Recent Momentum.
    """
    try:
        # Fetch detailed data (hourly if possible for short term, or daily)
        data = yf.Ticker(symbol).history(period="1mo", interval="1d")
        if data is None or len(data) < 26:
            return 0, 0, "Insufficient Data"
            
        close = data['Close']
        
        # 1. RSI Calculation (14 period)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        
        # 2. Moving Averages (Trend)
        sma_short = close.rolling(window=7).mean().iloc[-1]
        sma_long = close.rolling(window=25).mean().iloc[-1]
        
        # 3. Momentum (Last 3 days)
        momentum = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4] * 100
        
        score = 0
        reason = []
        
        # RSI Logic
        if rsi < 30: 
            score += 3
            reason.append("Oversold")
        elif rsi > 70: 
            score -= 3
            reason.append("Overbought")
        else:
            pass # Neutral RSI
            
        # Trend Logic
        if sma_short > sma_long:
            score += 4
            reason.append("Bullish Trend")
        else:
            score -= 4
            reason.append("Bearish Trend")
            
        # Momentum Logic
        if momentum > 5:
            score += 2
            reason.append("High Momentum")
        elif momentum < -5:
            score -= 1 
            
        return score, rsi, ", ".join(reason)
    except Exception as e:
        return 0, 0, f"Error: {str(e)}"

# ---------------------------
# TELEGRAM HELPER
# ---------------------------
def send_telegram_alert(message):
    token = st.session_state.get("tg_token")
    chat_id = st.session_state.get("tg_chat_id")
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        except Exception as e:
            print(f"Telegram Error: {e}")

# ---------------------------
# STATE MANAGEMENT
# ---------------------------
if "grid_bot_active" not in st.session_state:
    st.session_state["grid_bot_active"] = {} 

if "usd_inr" not in st.session_state:
    st.session_state["usd_inr"] = get_usd_inr_rate()

if "autopilot" not in st.session_state:
    st.session_state["autopilot"] = {
        "running": False, "mode": "PAPER", "currency": "USDT",
        "total_capital": 0.0, "cash_balance": 0.0,
        "active_grids": [], "logs": [], "history": [],
        "last_tg_update": dt.datetime.now() - dt.timedelta(hours=5) # Init to allow immediate update
    }
else:
    # Ensure all keys exist
    defaults = {
        "running": False, "mode": "PAPER", "currency": "USDT",
        "total_capital": 0.0, "cash_balance": 0.0,
        "active_grids": [], "logs": [], "history": [],
        "last_tg_update": dt.datetime.now() - dt.timedelta(hours=5)
    }
    for k, v in defaults.items():
        if k not in st.session_state["autopilot"]:
            st.session_state["autopilot"][k] = v

keys_to_init = ["crypto_running", "crypto_status", "crypto_loop_started",
                "binance_api", "binance_secret", "tg_token", "tg_chat_id"]

for key in keys_to_init:
    if key not in st.session_state:
        st.session_state[key] = "" if "tg" in key else (None if "binance" in key else False)

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
# PAGE 1: CRYPTO BOT (GRID TRADING)
# ---------------------------
def show_crypto_manual_bot_page():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="grid_refresh") 
    usd_inr = st.session_state["usd_inr"]

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
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD, key="bot_coin_select")
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.4f}")
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
        lower_p = col_a.number_input("Lower Price", value=st.session_state.get('auto_lower', 0.0), format="%.4f")
        upper_p = col_b.number_input("Upper Price", value=st.session_state.get('auto_upper', 0.0), format="%.4f")
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
            c2.write(f"${data['entry_price']:.4f}")
            c3.write(f"${cp:.4f}")
            c4.write(f"${data['invest']:.0f} / ₹{inv_inr:,.0f}")
            pnl_color = "green" if pnl_usd >= 0 else "red"
            c5.markdown(f":{pnl_color}[${pnl_usd:.2f} / ₹{pnl_inr:,.0f}]")
            c6.write(status_text)
            
            if c7.button("Stop", key=f"stop_{b_id}"):
                final_val_inr = inv_inr + pnl_inr
                msg = (f"🚨 *Manual Crypto Trade Closed*\n"
                       f"Asset: {data['coin']}\n"
                       f"💰 Invested: ₹{inv_inr:,.2f}\n"
                       f"💵 Final Value: ₹{final_val_inr:,.2f}\n"
                       f"📈 PnL: ₹{pnl_inr:,.2f} ({pnl_pct:.2f}%)")
                send_telegram_alert(msg)
                
                del st.session_state["grid_bot_active"][b_id]
                st.rerun()
    else:
        st.info("No active manual bots.")

    st.markdown("---")
    st.subheader(f"📉 Asset Price Chart: {selected_coin}")
    t_col1, t_col2 = st.columns([3, 1])
    with t_col1:
        time_range = st.radio("Select Timeframe", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd", "max"], index=2, horizontal=True)
    chart_data = get_safe_crypto_data(selected_coin, period=time_range)
    if chart_data is not None:
        fig = go.Figure(data=[go.Candlestick(x=chart_data.index, open=chart_data['Open'], high=chart_data['High'], low=chart_data['Low'], close=chart_data['Close'])])
        fig.update_layout(height=500, margin=dict(l=0,r=0,t=0,b=0), plot_bgcolor='black', paper_bgcolor='black', xaxis=dict(showgrid=True, gridcolor='#333', color='white'), yaxis=dict(showgrid=True, gridcolor='#333', color='white'), font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Chart data unavailable.")

# ---------------------------
# PAGE: AI AUTO-PILOT (ENHANCED ALGORITHM)
# ---------------------------
def show_ai_autopilot_page():
    usd_inr = st.session_state["usd_inr"]
    ap = st.session_state["autopilot"]
    api_key = st.session_state.get("binance_api")
    secret_key = st.session_state.get("binance_secret")
    is_live = bool(api_key and secret_key)
    
    st.title(f"🚀 AI Auto-Pilot ({'🔴 LIVE' if is_live else '🟢 PAPER'})")
    if is_live: st.warning("⚠️ LIVE MONEY MODE ACTIVE")
    else: st.info("ℹ️ Simulation Mode")
    st_autorefresh(interval=20_000, key="autopilot_refresh") 
    
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
            ap["last_tg_update"] = dt.datetime.now()
            ap["logs"].append(f"[{dt.datetime.now().strftime('%H:%M')}] AI Engine Started. Analyzing charts...")
            send_telegram_alert("🤖 *AI Auto-Pilot Started*\nMarket scanning initialized.")
            st.rerun()
    else:
        st.success(f"✅ AI Engine Running: Scanning BTC, ETH, SOL, BNB, ADA... (Updated: {dt.datetime.now(IST).strftime('%H:%M:%S')})")
        
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
        m2.metric("Cash", f"{curr_sym}{ap['cash_balance'] * conv_factor:,.2f}")
        m3.metric("Invested", f"{curr_sym}{grid_current_val * conv_factor:,.2f}")
        m4.metric("Net PnL", f"{curr_sym}{total_pnl_usd * conv_factor:,.2f}", delta=f"{(total_pnl_usd/ap['total_capital'])*100:.2f}%")
        
        st.markdown("---")
        
        # ==========================================
        # 1. AI JUDGE: EXIT LOGIC & MONITORING
        # ==========================================
        # Iterate backwards to safely remove items
        for i in range(len(ap['active_grids']) - 1, -1, -1):
            g = ap['active_grids'][i]
            cp = get_current_price(g['coin'])
            curr_val = g['qty'] * cp
            pnl = curr_val - g['invest']
            pnl_pct = (pnl / g['invest']) * 100
            
            # Re-Evaluate AI Score
            score, _, reason = calculate_ai_score(g['coin'])
            
            # Conditions for Bot to Stop Trade
            close_trade = False
            close_reason = ""
            
            # A. Hard Targets
            if pnl_pct >= g['tp']:
                close_trade = True
                close_reason = "✅ Take Profit Hit"
            elif pnl_pct <= -g['sl']:
                close_trade = True
                close_reason = "❌ Stop Loss Hit"
            
            # B. AI Sentiment Shift (The "Judge")
            # If we are long, and score becomes negative (Bearish), exit early to save capital
            elif score < -2: 
                close_trade = True
                close_reason = f"📉 AI Signal Reversal (Score: {score})"
            
            if close_trade:
                ap['cash_balance'] += curr_val
                ap['history'].append({"date": dt.datetime.now(IST), "pnl": pnl, "invested": g['invest'], "return_pct": pnl_pct})
                ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] {close_reason}: Closed {g['coin']}")
                
                # Telegram Alert: STOP
                inv_inr = g['invest'] * usd_inr
                pnl_inr = pnl * usd_inr
                msg = (f"🚨 *AI Trade Stopped ({close_reason})*\n"
                       f"Asset: {g['coin']}\n"
                       f"💰 PnL: ${pnl:.2f} (₹{pnl_inr:,.0f})\n"
                       f"📊 Return: {pnl_pct:.2f}%")
                send_telegram_alert(msg)
                
                ap['active_grids'].pop(i)
        
        # ==========================================
        # 2. AI JUDGE: ENTRY LOGIC
        # ==========================================
        # Scan if we have cash > 20% and not too many bots
        if ap['cash_balance'] > (ap['total_capital'] * 0.2): 
            best_score = -100
            best_coin = None
            best_reason = ""
            
            for coin in CRYPTO_SYMBOLS_USD:
                if any(g['coin'] == coin for g in ap['active_grids']): continue
                    
                score, rsi, reason = calculate_ai_score(coin)
                if score > best_score:
                    best_score = score
                    best_coin = coin
                    best_reason = reason
            
            # Entry Threshold
            if best_coin and best_score >= 4:
                cp = get_current_price(best_coin)
                if cp > 0:
                    allocation_pct = 0.4 if best_score >= 7 else 0.2
                    invest_amt = ap['cash_balance'] * allocation_pct
                    
                    # Cap investment if we want to reserve cash for a second bot
                    if ap['cash_balance'] >= (ap['total_capital'] * 0.5):
                        invest_amt = min(invest_amt, ap['cash_balance'] * 0.5)

                    lower = cp * 0.95
                    upper = cp * 1.05
                    tp_target = 2.5 if best_score >= 7 else 1.5
                    sl_target = 2.0
                    
                    grid_orders = [] # Mock orders for visual
                    for lvl in np.linspace(lower, upper, 5):
                        if lvl < cp: grid_orders.append({"type": "BUY", "price": lvl, "status": "OPEN"})
                        else: grid_orders.append({"type": "SELL", "price": lvl, "status": "OPEN"})

                    new_grid = {
                        "coin": best_coin, "entry": cp, 
                        "lower": lower, "upper": upper,
                        "qty": invest_amt/cp, "invest": invest_amt, 
                        "grids": 5, "tp": tp_target, "sl": sl_target,
                        "orders": grid_orders,
                        "ai_reason": best_reason,
                        "expected_profit": invest_amt * (tp_target/100),
                        "expected_loss": invest_amt * (sl_target/100)
                    }
                    ap['active_grids'].append(new_grid)
                    ap['cash_balance'] -= invest_amt
                    ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🤖 AI BUY {best_coin}: {best_reason}")
                    
                    # Telegram Alert: START
                    inv_inr = invest_amt * usd_inr
                    msg = (f"🚀 *AI Trade Started*\n"
                           f"Asset: {best_coin}\n"
                           f"reason: {best_reason}\n"
                           f"💰 Invested: ${invest_amt:.0f} (₹{inv_inr:,.0f})\n"
                           f"🎯 Target: +{tp_target}%")
                    send_telegram_alert(msg)

        # ==========================================
        # 3. TELEGRAM PERIODIC PNL REPORT
        # ==========================================
        # Send update every 4 hours OR if user forces it
        now = dt.datetime.now()
        last_update = ap.get('last_tg_update', now)
        if isinstance(last_update, str): last_update = pd.to_datetime(last_update) # Safety fix
        
        # Interval check (4 hours = 14400 seconds)
        if (now - last_update).total_seconds() > 14400:
            if ap['active_grids']:
                msg_lines = ["📊 *AI Auto-Pilot Periodic Report*"]
                total_curr_pnl = 0.0
                for g in ap['active_grids']:
                     cp = get_current_price(g['coin'])
                     pnl = (g['qty'] * cp) - g['invest']
                     total_curr_pnl += pnl
                     msg_lines.append(f"• {g['coin']}: ${pnl:.2f}")
                
                total_pnl_inr = total_curr_pnl * usd_inr
                msg_lines.append(f"\n💰 Net Floating PnL: ${total_curr_pnl:.2f} (₹{total_pnl_inr:,.0f})")
                send_telegram_alert("\n".join(msg_lines))
                ap['last_tg_update'] = now
                ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 📤 Sent Periodic Telegram Report")

        # ==========================================
        # UI DISPLAY
        # ==========================================
        st.subheader("🧠 AI Activity Log")
        for log in ap['logs'][:5]:
            st.text(log)

        st.markdown("---")

        st.subheader(f"💼 Active AI Trades")
        if ap['active_grids']:
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1, 1, 1, 1, 1, 1, 0.8])
            c1.markdown("**Asset**"); c2.markdown("**Score/Reason**"); 
            c3.markdown("**Invested**"); c4.markdown("**Exp. Profit**"); c5.markdown("**Exp. Loss**"); 
            c6.markdown("**Current Val**"); c7.markdown("**PnL**"); c8.markdown("**Action**")
            
            sum_inv = 0.0; sum_val = 0.0; sum_pnl = 0.0

            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                sum_inv += g['invest']; sum_val += curr_val; sum_pnl += pnl
                
                # Live Score
                score, _, _ = calculate_ai_score(g['coin'])

                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1, 1, 1, 1, 1, 1, 0.8])
                c1.write(g['coin'].replace("-USD",""))
                c2.caption(f"Score: {score} | {g.get('ai_reason','Auto')[:15]}...")
                c3.write(f"${g['invest']:.2f}")
                c4.markdown(f":green[+${g.get('expected_profit', 0):.2f}]")
                c5.markdown(f":red[-${g.get('expected_loss', 0):.2f}]")
                c6.write(f"${curr_val:.2f}")
                c7.markdown(f":{'green' if pnl>=0 else 'red'}[${pnl:.2f}]")
                
                if c8.button("Stop 🟥", key=f"ap_stop_{i}"):
                    ap['cash_balance'] += curr_val
                    pnl_pct = (pnl/g['invest'])*100
                    ap['history'].append({"date": dt.datetime.now(IST), "pnl": pnl, "invested": g['invest'], "return_pct": pnl_pct})
                    ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🛑 User Manual Stop: {g['coin']}")
                    
                    inv_inr = g['invest'] * usd_inr
                    pnl_inr = pnl * usd_inr
                    msg = (f"🚨 *Manual Stop Initiated*\n"
                           f"Asset: {g['coin']}\n"
                           f"💰 PnL: ${pnl:.2f} (₹{pnl_inr:,.0f})")
                    send_telegram_alert(msg)
                    
                    ap['active_grids'].pop(i)
                    st.rerun()
            
            st.markdown("<div style='border-top:1px solid #ddd; padding-top:10px; margin-top:5px;'></div>", unsafe_allow_html=True)
            f1, f2, f3, f4, f5, f6, f7, f8 = st.columns([1, 1, 1, 1, 1, 1, 1, 0.8])
            f1.write("**TOTALS**")
            f3.write(f"**${sum_inv:,.2f}**")
            f6.write(f"**${sum_val:,.2f}**")
            pnl_col = "green" if sum_pnl >= 0 else "red"
            f7.markdown(f":{pnl_col}[**${sum_pnl:,.2f}**]")

            # --- GRID DETAILS SECTION ADDED BACK HERE ---
            st.markdown("---")
            st.markdown("### 📋 AI Grid Orders")
            for g in ap['active_grids']:
                with st.expander(f"Orders for {g['coin'].replace('-USD','')}"):
                    if 'orders' in g and g['orders']:
                        ord_df = pd.DataFrame(g['orders'])
                        ord_df['price'] = ord_df['price'].apply(lambda x: f"${x:.4f}")
                        st.table(ord_df)
                    else:
                        st.write("Generating orders...")
            # --------------------------------------------

        else:
            st.info("AI Scanner Active: Waiting for optimal setup...")
        
        if st.button("⏹ Stop AI Engine"): 
            ap["running"] = False
            send_telegram_alert("🛑 *AI Auto-Pilot Stopped manually.*")
            st.rerun()

# ---------------------------
# PAGE: CRYPTO REPORT
# ---------------------------
def show_crypto_report_page():
    st.title("📑 Crypto PnL Report")
    ap = st.session_state["autopilot"]
    usd_inr = st.session_state["usd_inr"]
    
    st_autorefresh(interval=30_000, key="report_refresh")

    st.subheader("🔴 Live Portfolio")
    running_inv = sum([g.get('invest', 0.0) for g in ap['active_grids']])
    running_val = sum([(g['qty'] * get_current_price(g['coin'])) for g in ap['active_grids']])
    
    total_exp_profit = sum([g.get('expected_profit', 0.0) for g in ap['active_grids']])
    total_exp_loss = sum([g.get('expected_loss', 0.0) for g in ap['active_grids']])

    c1, c2, c3 = st.columns(3)
    c1.metric("Live Invested", f"₹{running_inv * usd_inr:,.0f}")
    c2.metric("Live Value", f"₹{running_val * usd_inr:,.0f}")
    c3.metric("Live PnL", f"₹{(running_val - running_inv) * usd_inr:,.0f}")

    c4, c5 = st.columns(2)
    c4.metric("📈 Total Expected Profit", f"₹{total_exp_profit * usd_inr:,.0f} (${total_exp_profit:.2f})")
    c5.metric("📉 Total Expected Loss (Risk)", f"₹{total_exp_loss * usd_inr:,.0f} (${total_exp_loss:.2f})")
    
    st.markdown("---")
    st.subheader("🏁 Closed Trades")
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        df['date'] = pd.to_datetime(df['date'])
        total_profit = df['pnl'].sum()
        m1, m2 = st.columns(2)
        m1.metric("Realized PnL", f"${total_profit:,.2f} (₹{total_profit*usd_inr:,.0f})")
        m2.metric("Trades", len
