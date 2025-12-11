import os
import time
import threading
import datetime as dt
import sqlite3
from typing import Dict, List, Optional
import random
import json

import pytz
import requests
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import yfinance as yf
import plotly.graph_objects as go

# --- IMPORT GOOGLE GEMINI ---
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

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
        
        /* --- SIDEBAR INPUTS --- */
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            caret-color: #ffffff !important;
            border: 1px solid #666 !important;
        }
        section[data-testid="stSidebar"] label { color: #ffffff !important; }
        
        /* --- EXPANDER & TABLE FIXES (WHITE THEME) --- */
        /* Force Expander Header to be Light Grey */
        .main div[data-testid="stExpander"] details summary {
            background-color: #f0f2f6 !important;
            color: #000000 !important;
            border: 1px solid #dee2e6;
        }
        
        /* Force Tables/Dataframes inside Expander to be WHITE with BLACK TEXT */
        .main div[data-testid="stExpander"] div[data-testid="stDataFrame"] {
            background-color: #ffffff !important;
        }
        .main div[data-testid="stExpander"] div[data-testid="stDataFrame"] div,
        .main div[data-testid="stExpander"] div[data-testid="stDataFrame"] span,
        .main div[data-testid="stExpander"] table,
        .main div[data-testid="stExpander"] th,
        .main div[data-testid="stExpander"] td {
            color: #000000 !important;
            background-color: #ffffff !important;
        }
        
        /* --- DROPDOWN MENU FIX --- */
        div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ced4da;
        }
        div[data-baseweb="popover"], div[data-baseweb="menu"], div[role="option"] {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        
        /* Metrics */
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
    return 84.0 

# ---------------------------
# 🧠 GEMINI 3 INTELLIGENT ENGINE
# ---------------------------
def gemini3_analysis(symbol, api_key):
    """
    Uses Google Gemini to analyze market data and output a trading decision.
    """
    if not GEMINI_AVAILABLE or not api_key:
        return calculate_technical_score(symbol) # Fallback

    try:
        # 1. Fetch Data
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        if hist.empty: return 0, 0, "No Data"
        
        current_price = hist['Close'].iloc[-1]
        prices_str = ", ".join([f"${p:.2f}" for p in hist['Close'].tolist()])
        
        # 2. Construct Prompt
        prompt = f"""
        You are an expert crypto trading AI (Gemini 3 Engine).
        Analyze {symbol}. Recent 5 daily closing prices: [{prices_str}].
        Current Price: ${current_price:.2f}.
        
        Task: Provide a Trading Score from -10 (Strong Sell) to +10 (Strong Buy).
        Output STRICT JSON format only: {{"score": int, "reason": "short string"}}
        """
        
        # 3. Call API
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        # 4. Parse
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return int(data.get("score", 0)), 50, f"Gemini: {data.get('reason', 'Analysis')}"
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        return calculate_technical_score(symbol) # Fallback on error

def calculate_technical_score(symbol):
    """Fallback Standard Algo"""
    try:
        data = yf.Ticker(symbol).history(period="1mo", interval="1d")
        if data is None or len(data) < 26: return 0, 0, "Insufficient Data"
        close = data['Close']
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        sma_short = close.rolling(window=7).mean().iloc[-1]
        sma_long = close.rolling(window=25).mean().iloc[-1]
        
        score = 0
        reason = []
        if rsi < 30: score += 3; reason.append("Oversold")
        elif rsi > 70: score -= 3; reason.append("Overbought")
        if sma_short > sma_long: score += 4; reason.append("Bullish Trend")
        else: score -= 4; reason.append("Bearish Trend")
        
        return score, rsi, ", ".join(reason)
    except:
        return 0, 0, "Error"

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
if "grid_bot_active" not in st.session_state: st.session_state["grid_bot_active"] = {} 
if "usd_inr" not in st.session_state: st.session_state["usd_inr"] = get_usd_inr_rate()

if "autopilot" not in st.session_state:
    st.session_state["autopilot"] = {
        "running": False, "mode": "PAPER", "currency": "USDT",
        "total_capital": 0.0, "cash_balance": 0.0,
        "active_grids": [], "logs": [], "history": [],
        "last_tg_update": dt.datetime.now() - dt.timedelta(hours=5)
    }

keys_to_init = ["binance_api", "binance_secret", "tg_token", "tg_chat_id", "gemini_api_key"]
for key in keys_to_init:
    if key not in st.session_state: st.session_state[key] = ""

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
# PAGE: AI AUTO-PILOT
# ---------------------------
def show_ai_autopilot_page():
    usd_inr = st.session_state["usd_inr"]
    ap = st.session_state["autopilot"]
    gemini_key = st.session_state.get("gemini_api_key")
    
    st.title(f"🚀 AI Auto-Pilot")
    
    # Engine Status Label
    if gemini_key: st.caption("✨ Powered by Gemini 3 Intelligent Engine")
    else: st.caption("⚙️ Running on Standard Technical Engine")

    st_autorefresh(interval=20_000, key="autopilot_refresh") 
    
    if not ap["running"]:
        st.subheader("🛠️ Setup Auto-Pilot")
        c1, c2, c3 = st.columns(3)
        currency_mode = c1.radio("Capital Currency", ["USDT (USD)", "INR (₹)"])
        capital_input = c2.number_input("Total Capital Allocation", min_value=10.0, value=1000.0, step=100.0)
        
        if c3.button("🚀 Launch AI", type="primary", use_container_width=True):
            ap["running"] = True
            ap["mode"] = "PAPER"
            ap["currency"] = "USDT" if "USDT" in currency_mode else "INR"
            if ap["currency"] == "INR":
                ap["total_capital"] = capital_input / usd_inr
                ap["cash_balance"] = capital_input / usd_inr
            else:
                ap["total_capital"] = capital_input
                ap["cash_balance"] = capital_input
            ap["active_grids"] = []
            ap["last_tg_update"] = dt.datetime.now()
            ap["logs"].append(f"[{dt.datetime.now().strftime('%H:%M')}] AI Engine Started.")
            send_telegram_alert("🤖 *AI Auto-Pilot Started*")
            st.rerun()
    else:
        st.success(f"✅ AI Engine Running (Updated: {dt.datetime.now(IST).strftime('%H:%M:%S')})")
        
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
        # 1. AI JUDGE: EXIT LOGIC
        # ==========================================
        for i in range(len(ap['active_grids']) - 1, -1, -1):
            g = ap['active_grids'][i]
            cp = get_current_price(g['coin'])
            curr_val = g['qty'] * cp
            pnl = curr_val - g['invest']
            pnl_pct = (pnl / g['invest']) * 100
            
            # Use Gemini or Standard
            score, _, reason = gemini3_analysis(g['coin'], gemini_key)
            
            close_trade = False
            close_reason = ""
            
            if pnl_pct >= g['tp']:
                close_trade = True; close_reason = "✅ Take Profit Hit"
            elif pnl_pct <= -g['sl']:
                close_trade = True; close_reason = "❌ Stop Loss Hit"
            elif score < -3: 
                close_trade = True; close_reason = f"📉 AI Signal Reversal ({reason})"
            
            if close_trade:
                ap['cash_balance'] += curr_val
                ap['history'].append({"date": dt.datetime.now(IST), "pnl": pnl, "invested": g['invest'], "return_pct": pnl_pct})
                ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] {close_reason}: Closed {g['coin']}")
                msg = (f"🚨 *AI Trade Stopped ({close_reason})*\nAsset: {g['coin']}\n💰 PnL: ${pnl:.2f} ({pnl_pct:.2f}%)")
                send_telegram_alert(msg)
                ap['active_grids'].pop(i)
        
        # ==========================================
        # 2. AI JUDGE: ENTRY LOGIC
        # ==========================================
        if ap['cash_balance'] > (ap['total_capital'] * 0.2): 
            best_score = -100; best_coin = None; best_reason = ""
            
            for coin in CRYPTO_SYMBOLS_USD:
                if any(g['coin'] == coin for g in ap['active_grids']): continue
                
                # Analyze with Gemini 3
                score, _, reason = gemini3_analysis(coin, gemini_key)
                
                if score > best_score:
                    best_score = score; best_coin = coin; best_reason = reason
            
            if best_coin and best_score >= 5:
                cp = get_current_price(best_coin)
                if cp > 0:
                    alloc = 0.4 if best_score >= 8 else 0.2
                    invest_amt = min(ap['cash_balance'] * alloc, ap['cash_balance'] * 0.5)

                    lower = cp * 0.95; upper = cp * 1.05
                    tp_t = 3.0 if best_score >= 8 else 1.5; sl_t = 2.0
                    
                    grid_orders = []
                    for lvl in np.linspace(lower, upper, 5):
                        if lvl < cp: grid_orders.append({"type": "BUY", "price": lvl, "status": "OPEN"})
                        else: grid_orders.append({"type": "SELL", "price": lvl, "status": "OPEN"})

                    new_grid = {
                        "coin": best_coin, "entry": cp, 
                        "lower": lower, "upper": upper,
                        "qty": invest_amt/cp, "invest": invest_amt, 
                        "grids": 5, "tp": tp_t, "sl": sl_t,
                        "orders": grid_orders,
                        "ai_reason": best_reason,
                        "expected_profit": invest_amt * (tp_t/100),
                        "expected_loss": invest_amt * (sl_t/100)
                    }
                    ap['active_grids'].append(new_grid)
                    ap['cash_balance'] -= invest_amt
                    ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🤖 AI BUY {best_coin}: {best_reason}")
                    
                    inv_inr = invest_amt * usd_inr
                    msg = (f"🚀 *AI Trade Started*\nAsset: {best_coin}\nEngine: {'Gemini 3' if gemini_key else 'Standard'}\nReason: {best_reason}\n💰 Invested: ${invest_amt:.0f} (₹{inv_inr:,.0f})")
                    send_telegram_alert(msg)

        # ==========================================
        # 3. REPORTING
        # ==========================================
        now = dt.datetime.now()
        last_update = ap.get('last_tg_update', now)
        if isinstance(last_update, str): last_update = pd.to_datetime(last_update)
        
        if (now - last_update).total_seconds() > 14400 and ap['active_grids']:
             msg_lines = ["📊 *Periodic AI Report*"]
             for g in ap['active_grids']:
                  curr_pnl = (g['qty'] * get_current_price(g['coin'])) - g['invest']
                  msg_lines.append(f"• {g['coin']}: ${curr_pnl:.2f}")
             send_telegram_alert("\n".join(msg_lines))
             ap['last_tg_update'] = now

        # ==========================================
        # UI DISPLAY
        # ==========================================
        st.subheader("🧠 AI Activity Log")
        for log in ap['logs'][:5]: st.text(log)
        st.markdown("---")

        st.subheader(f"💼 Active AI Trades")
        if ap['active_grids']:
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1, 1, 1, 1, 1, 1, 0.8])
            c1.markdown("**Asset**"); c2.markdown("**Score/Reason**"); 
            c3.markdown("**Invested**"); c4.markdown("**Exp. Profit**"); c5.markdown("**Exp. Loss**"); 
            c6.markdown("**Current Val**"); c7.markdown("**PnL**"); c8.markdown("**Action**")
            
            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                
                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1, 1, 1, 1, 1, 1, 0.8])
                c1.write(g['coin'].replace("-USD",""))
                c2.caption(f"{g.get('ai_reason','Auto')[:15]}...")
                c3.write(f"${g['invest']:.2f}")
                c4.markdown(f":green[+${g.get('expected_profit', 0):.2f}]")
                c5.markdown(f":red[-${g.get('expected_loss', 0):.2f}]")
                c6.write(f"${curr_val:.2f}")
                c7.markdown(f":{'green' if pnl>=0 else 'red'}[${pnl:.2f}]")
                
                if c8.button("Stop 🟥", key=f"stop_{i}"):
                    ap['cash_balance'] += curr_val
                    ap['active_grids'].pop(i)
                    st.rerun()

            st.markdown("---")
            st.markdown("### 📋 AI Grid Orders")
            for g in ap['active_grids']:
                with st.expander(f"Orders for {g['coin'].replace('-USD','')} (Click to View)"):
                    if 'orders' in g and g['orders']:
                        ord_df = pd.DataFrame(g['orders'])
                        ord_df['price'] = ord_df['price'].apply(lambda x: f"${x:.4f}")
                        # Table is now styled WHITE in CSS
                        st.dataframe(ord_df, use_container_width=True)
                    else:
                        st.write("Generating orders...")
        else:
            st.info("AI Scanner Active: Waiting for optimal setup...")
        
        if st.button("⏹ Stop AI Engine"): 
            ap["running"] = False
            st.rerun()

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    st.sidebar.title("💰 Paisa Banao")
    st.sidebar.title("Navigation")
    
    current_page = st.sidebar.radio("Menu", ["AI Auto-Pilot", "Crypto Report", "Manual Bot"], label_visibility="collapsed")
    st.sidebar.markdown("---")
    
    with st.sidebar.expander("🤖 Gemini AI Config"):
        gemini_key = st.text_input("Gemini API Key", value=st.session_state.get("gemini_api_key", ""), type="password")
        if st.button("Save API Key"):
            st.session_state["gemini_api_key"] = gemini_key
            st.success("Saved!")

    with st.sidebar.expander("📢 Telegram Alerts"):
        tg_token = st.text_input("Bot Token", value=st.session_state.get("tg_token", ""), type="password")
        tg_chat = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))
        if st.button("Save Telegram"):
            st.session_state["tg_token"] = tg_token
            st.session_state["tg_chat_id"] = tg_chat
            st.success("Saved!")

    if CRYPTO_BOT_AVAILABLE and not st.session_state.get("crypto_loop_started", False):
        t_crypto = threading.Thread(target=crypto_trading_loop, daemon=True)
        t_crypto.start()
        st.session_state["crypto_loop_started"] = True

    if current_page == "AI Auto-Pilot": show_ai_autopilot_page()
    elif current_page == "Crypto Report": 
        # Placeholder for report page
        st.title("Report")
        st.write("Use Main Page for details")
    elif current_page == "Manual Bot": show_crypto_manual_bot_page()

def show_crypto_manual_bot_page():
    st.title("Manual Bot")
    st.write("Use AI Auto-Pilot for smart trading.")

if __name__ == "__main__":
    main()
