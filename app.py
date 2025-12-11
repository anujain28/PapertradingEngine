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
import plotly.express as px

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
# 🔐 USER AUTHENTICATION CONFIG
# ---------------------------
USER_CREDENTIALS = {
    "username": "admin",
    "password": "admin"
}

# ---------------------------
# PAGE CONFIG + GLOBAL STYLE
# ---------------------------
st.set_page_config(page_title="AI Crypto Engine", layout="wide", page_icon="📈")

def apply_custom_style():
    st.markdown("""
        <style>
        /* Global App Background */
        .stApp { background-color: #ffffff !important; color: #000000 !important; }
        p, h1, h2, h3, h4, h5, h6, span, div, label, li, td, th { color: #000000 !important; }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] { background-color: #262730 !important; color: white !important; }
        section[data-testid="stSidebar"] * { color: white !important; }
        section[data-testid="stSidebar"] input { 
            background-color: #000000 !important; 
            color: #ffffff !important; 
            border: 1px solid #666 !important;
        }
        
        /* Main Page Inputs */
        .main input { 
            background-color: #ffffff !important; 
            color: #000000 !important; 
            border: 1px solid #ccc !important; 
        }
        
        /* Expanders & Dataframes */
        .main div[data-testid="stExpander"] details summary { 
            background-color: #f0f2f6 !important; 
            color: #000000 !important; 
            border: 1px solid #dee2e6; 
        }
        .main div[data-testid="stExpander"] div[role="group"] { 
            background-color: #ffffff !important; 
        }
        div[data-testid="stDataFrame"] { 
            background-color: #ffffff !important; 
        }
        div[data-testid="stDataFrame"] * { 
            color: #000000 !important; 
        }
        
        /* Metrics Box */
        div[data-testid="metric-container"] {
            background-color: #f8f9fa !important;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 15px;
            color: #000000 !important;
        }
        div[data-testid="metric-container"] label { color: #555 !important; }
        
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
DB_PATH = "paper_trades.db"
CRYPTO_SYMBOLS_USD = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]

# ---------------------------
# 🧠 GLOBAL SHARED STATE (PERSISTENT ENGINE)
# ---------------------------
class SharedEngineState:
    def __init__(self):
        self.autopilot = {
            "running": False, 
            "mode": "PAPER", 
            "currency": "USDT",
            "total_capital": 0.0, 
            "cash_balance": 0.0,
            "active_grids": [], 
            "logs": [], 
            "history": [],
            "last_tg_update": dt.datetime.now() - dt.timedelta(hours=5)
        }
        self.grid_bot_active = {}
        self.gemini_api_key = ""
        self.last_gemini_usage = None
        self.tg_token = ""
        self.tg_chat_id = ""
        self.binance_api = ""
        self.binance_secret = ""

@st.cache_resource
def get_engine():
    return SharedEngineState()

ENGINE = get_engine()

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
        if not data.empty: return data["Close"].iloc[-1]
    except: pass
    return 84.0 

# ---------------------------
# 🧠 GEMINI 3 INTELLIGENT ENGINE
# ---------------------------
def check_gemini_eligibility():
    last_use_str = ENGINE.last_gemini_usage
    if last_use_str is None: return True 
    try:
        last_use = dt.datetime.strptime(last_use_str, "%Y-%m-%d %H:%M:%S")
        if (dt.datetime.now() - last_use).days >= 3: return True
        return False
    except: return True

def gemini3_analysis(symbol):
    api_key = ENGINE.gemini_api_key
    if not api_key: return calculate_technical_score(symbol)
    if not GEMINI_AVAILABLE: return calculate_technical_score(symbol)
    if not check_gemini_eligibility(): return calculate_technical_score(symbol)

    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        if hist.empty: return 0, 0, "No Data"
        
        current_price = hist['Close'].iloc[-1]
        prices_str = ", ".join([f"${p:.2f}" for p in hist['Close'].tolist()])
        
        prompt = f"""
        You are an expert crypto trading AI (Gemini 3 Engine).
        Analyze {symbol}. Recent 5 daily closing prices: [{prices_str}].
        Current Price: ${current_price:.2f}.
        Task: Provide a Trading Score from -10 (Strong Sell) to +10 (Strong Buy).
        Output STRICT JSON format only: {{"score": int, "reason": "short string"}}
        """
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        clean_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        
        ENGINE.last_gemini_usage = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return int(data.get("score", 0)), 50, f"Gemini: {data.get('reason', 'Analysis')}"
    except Exception as e:
        return calculate_technical_score(symbol)

def calculate_technical_score(symbol):
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

def send_telegram_alert(message):
    token = ENGINE.tg_token
    chat_id = ENGINE.tg_chat_id
    if token and chat_id:
        try:
            url = f"https://api.telegram.org/bot{token}/sendMessage"
            requests.post(url, json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})
        except Exception as e:
            print(f"Telegram Error: {e}")

# ---------------------------
# DATABASE & BACKGROUND THREAD
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS trades (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, side TEXT, qty INTEGER, price REAL, timestamp TEXT, pnl REAL)""")
    conn.commit()
    conn.close()

def engine_background_logic():
    """Runs permanently in the background."""
    while True:
        # 1. AI AUTO-PILOT LOGIC
        ap = ENGINE.autopilot
        if ap["running"]:
            for i in range(len(ap['active_grids']) - 1, -1, -1):
                g = ap['active_grids'][i]
                cp = get_current_price(g['coin'])
                pnl_pct = ((g['qty'] * cp) - g['invest']) / g['invest'] * 100
                
                close = False
                if pnl_pct >= g['tp']: close = True
                elif pnl_pct <= -g['sl']: close = True
                
                if close:
                    ap['cash_balance'] += (g['qty'] * cp)
                    ap['active_grids'].pop(i)
                    # Log PnL to history
                    ap['history'].append({
                        "date": dt.datetime.now(IST), 
                        "coin": g['coin'],
                        "pnl": (g['qty'] * cp) - g['invest'], 
                        "invested": g['invest'], 
                        "return_pct": pnl_pct
                    })
                    send_telegram_alert(f"🚨 Engine Closed {g['coin']} at {pnl_pct:.2f}%")

        # 2. MANUAL BOT LOGIC
        manual_bots = ENGINE.grid_bot_active
        for b_id, data in list(manual_bots.items()):
            cp = get_current_price(data['coin'])
            current_val_usd = data['qty'] * cp
            pnl_usd = current_val_usd - data['invest']
            pnl_pct = (pnl_usd / data['invest']) * 100
            
            close = False
            if pnl_pct >= data['tp']: close = True
            elif pnl_pct <= -data['sl']: close = True
            
            if close:
                send_telegram_alert(f"🚨 Manual Bot Closed {data['coin']} at {pnl_pct:.2f}%")
                del manual_bots[b_id]

        time.sleep(60)

@st.cache_resource
def start_background_thread():
    t = threading.Thread(target=engine_background_logic, daemon=True)
    t.start()
    return t

# ---------------------------
# PAGES
# ---------------------------
def show_login_page():
    st.markdown("<h1 style='text-align: center; color:black;'>🔐 Secure Engine Login</h1>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Access Engine")
            
            if submitted:
                if username == USER_CREDENTIALS["username"] and password == USER_CREDENTIALS["password"]:
                    st.session_state["authenticated"] = True
                    st.success("Access Granted. Loading...")
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")

def show_ai_autopilot_page(usd_inr):
    ap = ENGINE.autopilot
    st.title(f"🚀 AI Auto-Pilot")
    
    is_gemini_restricted = not check_gemini_eligibility()
    if ENGINE.gemini_api_key and not is_gemini_restricted:
        st.caption("✨ Powered by Gemini 3 Intelligent Engine")
    else:
        st.caption("⚙️ Running on Standard Technical Engine")

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
        
        # 1. SCANNER ENTRY LOGIC (Frontend View)
        scan_data = []
        for coin in CRYPTO_SYMBOLS_USD:
            cp = get_current_price(coin)
            s_score, _, s_reason = gemini3_analysis(coin)
            
            status_msg = "Waiting"
            if any(g['coin'] == coin for g in ap['active_grids']): status_msg = "Active"
            elif s_score >= 5: status_msg = "Ready to Buy"
            elif s_score <= -3: status_msg = "Avoid (Bearish)"
            else: status_msg = "Neutral"
            
            scan_data.append({
                "Asset": coin.replace("-USD",""),
                "Price": f"${cp:.2f}",
                "AI Score": s_score,
                "Analysis": s_reason,
                "Status": status_msg
            })
            
        # Trigger Buy if valid (Simulated here for immediate feedback)
        if ap['cash_balance'] > (ap['total_capital'] * 0.2): 
            best_score = -100; best_coin = None; best_reason = ""
            for coin in CRYPTO_SYMBOLS_USD:
                if any(g['coin'] == coin for g in ap['active_grids']): continue
                score, _, reason = gemini3_analysis(coin)
                if score > best_score:
                    best_score = score; best_coin = coin; best_reason = reason
            
            if best_coin and best_score >= 5:
                cp = get_current_price(best_coin)
                if cp > 0:
                    alloc = 0.4 if best_score >= 8 else 0.2
                    invest_amt = min(ap['cash_balance'] * alloc, ap['cash_balance'] * 0.5)
                    hist = get_safe_crypto_data(best_coin, period="5d")
                    vol_pct = 0.05
                    if hist is not None:
                         vol_pct = max(0.03, ((hist['High'] - hist['Low']) / hist['Close']).mean() * 1.5)

                    lower = cp * (1 - vol_pct); upper = cp * (1 + vol_pct)
                    tp_t = round(vol_pct * 100 * 0.8, 2); sl_t = round(vol_pct * 100 * 0.5, 2)
                    grid_count = max(3, min(10, int(invest_amt / 15)))

                    grid_orders = []
                    for lvl in np.linspace(lower, upper, grid_count):
                        if lvl < cp: grid_orders.append({"type": "BUY", "price": lvl, "status": "OPEN"})
                        else: grid_orders.append({"type": "SELL", "price": lvl, "status": "OPEN"})

                    new_grid = {
                        "coin": best_coin, "entry": cp, 
                        "lower": lower, "upper": upper,
                        "qty": invest_amt/cp, "invest": invest_amt, 
                        "grids": grid_count, "tp": tp_t, "sl": sl_t,
                        "orders": grid_orders,
                        "ai_reason": best_reason,
                        "expected_profit": invest_amt * (tp_t/100),
                        "expected_loss": invest_amt * (sl_t/100)
                    }
                    ap['active_grids'].append(new_grid)
                    ap['cash_balance'] -= invest_amt
                    ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🤖 AI BUY {best_coin}: {best_reason}")
                    msg = (f"🚀 *AI Trade Started*\nAsset: {best_coin}\nReason: {best_reason}\n💰 Invested: ${invest_amt:.0f}")
                    send_telegram_alert(msg)

        # UI DISPLAY
        st.subheader("🔍 Live AI Scanner Insights")
        if scan_data: st.dataframe(pd.DataFrame(scan_data), use_container_width=True)
        else: st.info("Initializing Scanner...")
            
        st.markdown("---")
        st.subheader("🧠 AI Activity Log")
        for log in ap['logs'][:5]: st.text(log)
        st.markdown("---")

        st.subheader(f"💼 Active AI Trades")
        if ap['active_grids']:
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1, 1, 1, 1, 1, 1, 0.8])
            c1.markdown("**Asset**"); c2.markdown("**Reason**"); 
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
                        st.dataframe(ord_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Active Trade Configurations")
            config_data = []
            for g in ap['active_grids']:
                config_data.append({
                    "Asset": g['coin'],
                    "Lower Grid ($)": f"{g['lower']:.4f}",
                    "Upper Grid ($)": f"{g['upper']:.4f}",
                    "TP (%)": f"{g['tp']}%",
                    "SL (%)": f"{g['sl']}%",
                    "Total Grids": g.get('grids', 5),
                    "Invested (INR)": f"₹{g['invest'] * usd_inr:,.2f}"
                })
            st.table(pd.DataFrame(config_data))
        else:
            st.info("AI Scanner Active: Waiting for optimal setup...")
        
        if st.button("⏹ Stop AI Engine"): 
            ap["running"] = False
            st.rerun()

def show_crypto_manual_bot_page(usd_inr):
    st.title("🤖 AI Crypto Manual Bot")
    st.autorefresh(interval=30_000, key="grid_refresh")
    
    st.subheader("🔎 Live Market Analysis (USDT)")
    analysis_data = []
    for coin in CRYPTO_SYMBOLS_USD:
        cp = get_current_price(coin)
        analysis_data.append({"Coin": coin, "Price": f"${cp:.2f}"})
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    st.markdown("---")

    # Manual Config
    st.subheader("⚙️ Configure Manual Bot")
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD, key="bot_coin_select")
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.4f}")
        
    with c2:
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower Price", value=curr_price*0.95)
        upper_p = col_b.number_input("Upper Price", value=curr_price*1.05)
        col_c, col_d = st.columns(2)
        grids = col_c.number_input("Grids", min_value=2, max_value=20, value=5)
        invest = col_d.number_input("Investment ($)", value=100.0)
        col_e, col_f = st.columns(2)
        tp_pct = col_e.number_input("TP (%)", value=2.0)
        sl_pct = col_f.number_input("SL (%)", value=3.0)

    if st.button("▶️ Start Manual Bot"):
        if curr_price > 0:
            bot_id = selected_coin
            entry_qty = invest / curr_price
            ENGINE.grid_bot_active[bot_id] = {
                "coin": selected_coin, "entry_price": curr_price,
                "lower": lower_p, "upper": upper_p, "grids": grids,
                "qty": entry_qty, "invest": invest, "tp": tp_pct, "sl": sl_pct,
                "status": "Running", "start_time": dt.datetime.now().strftime("%H:%M:%S")
            }
            st.success("Bot Started!")

    st.markdown("---")
    st.subheader("📍 Active Manual Bots")
    active_bots = ENGINE.grid_bot_active
    if active_bots:
        h1, h2, h3, h4, h5, h6 = st.columns([1,1,1,2,2,1])
        h1.write("**Coin**"); h2.write("**Entry**"); h3.write("**CMP**"); 
        h4.write("**Inv**"); h5.write("**PnL**"); h6.write("**Action**")
        
        for b_id, data in list(active_bots.items()):
            cp = get_current_price(data['coin'])
            current_val_usd = data['qty'] * cp
            pnl_usd = current_val_usd - data['invest']
            pnl_pct = (pnl_usd / data['invest']) * 100
            
            c1, c2, c3, c4, c5, c6 = st.columns([1,1,1,2,2,1])
            c1.write(data['coin'])
            c2.write(f"${data['entry_price']:.2f}")
            c3.write(f"${cp:.2f}")
            c4.write(f"${data['invest']:.0f}")
            c5.markdown(f":{'green' if pnl_usd>=0 else 'red'}[${pnl_usd:.2f}]")
            if c6.button("Stop", key=f"stop_{b_id}"):
                del ENGINE.grid_bot_active[b_id]
                st.rerun()
    else:
        st.info("No active manual bots.")

def show_crypto_report_page(usd_inr):
    st.title("📑 Crypto PnL Report")
    ap = ENGINE.autopilot
    st_autorefresh(interval=30_000, key="report_refresh")
    
    # 1. METRICS
    total_profit = sum([t.get('pnl', 0) for t in ap['history']])
    st.metric("💰 Total Realized PnL", f"${total_profit:,.2f}", help="Total profit from closed trades")
    
    # 2. CHART (PnL Growth)
    st.subheader("📈 Profit Growth")
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        df['cumulative_pnl'] = df['pnl'].cumsum()
        
        fig = px.line(df, x='date', y='cumulative_pnl', markers=True, title="Cumulative PnL Over Time")
        fig.update_layout(xaxis_title="Time", yaxis_title="Profit (USD)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No closed trades to chart yet.")

    # 3. DETAILED TABLE
    st.subheader("📋 Trade History")
    if ap["history"]:
        df_display = pd.DataFrame(ap["history"])
        df_display['date'] = pd.to_datetime(df_display['date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(df_display.sort_values('date', ascending=False), use_container_width=True)
    else:
        st.info("No history available.")

# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    apply_custom_style()
    init_db()
    
    # 1. Start Persistent Engine (Runs ONCE per server start)
    start_background_thread()
    
    # 2. Auth Logic
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False

    if not st.session_state["authenticated"]:
        show_login_page()
    else:
        # LOGGED IN VIEW
        st.sidebar.title("💰 Paisa Banao")
        if st.sidebar.button("🔒 Logout"):
            st.session_state["authenticated"] = False
            st.rerun()
            
        current_page = st.sidebar.radio("Menu", ["AI Auto-Pilot", "Crypto Report", "Manual Bot"])
        st.sidebar.markdown("---")
        
        with st.sidebar.expander("🤖 Gemini AI Config"):
            gemini_key = st.text_input("Gemini API Key", value=ENGINE.gemini_api_key, type="password")
            if st.button("Save API Key"):
                ENGINE.gemini_api_key = gemini_key
                st.success("Saved!")

        with st.sidebar.expander("📢 Telegram Alerts"):
            tg_token = st.text_input("Bot Token", value=ENGINE.tg_token, type="password")
            tg_chat = st.text_input("Chat ID", value=ENGINE.tg_chat_id)
            if st.button("Save Telegram"):
                ENGINE.tg_token = tg_token
                ENGINE.tg_chat_id = tg_chat
                st.success("Saved!")

        usd_inr = get_usd_inr_rate()
        if current_page == "AI Auto-Pilot": show_ai_autopilot_page(usd_inr)
        elif current_page == "Crypto Report": show_crypto_report_page(usd_inr)
        elif current_page == "Manual Bot": show_crypto_manual_bot_page(usd_inr)

if __name__ == "__main__":
    main()
