import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import random
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from modules.utils import IST, send_telegram_alert, get_usd_inr_rate

CRYPTO_SYMBOLS_USD = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]

# --- HELPER FUNCTIONS ---
def get_current_price(symbol):
    try:
        data = yf.Ticker(symbol).history(period="1d")
        if not data.empty: return data["Close"].iloc[-1]
    except: pass
    return 0.0

@st.cache_data(ttl=300)
def get_safe_crypto_data(symbol, period="1mo"):
    try:
        return yf.Ticker(symbol).history(period=period)
    except: return None

# --- STATE INIT ---
def init_crypto_state():
    if "grid_bot_active" not in st.session_state:
        st.session_state["grid_bot_active"] = {} 
    
    if "autopilot" not in st.session_state:
        st.session_state["autopilot"] = {
            "running": False, "mode": "PAPER", "currency": "USDT",
            "total_capital": 0.0, "cash_balance": 0.0,
            "active_grids": [], "logs": [], "history": []
        }

# --- BACKGROUND TRADING ENGINE (THREAD) ---
def crypto_trading_loop():
    """
    Runs in the background 24/7 to manage Auto-Pilot grids 
    independently of the UI.
    """
    while True:
        try:
            # Run every 15 seconds to avoid API rate limits
            time.sleep(15)
            
            # Ensure we have access to state
            if "autopilot" not in st.session_state: continue
            
            ap = st.session_state["autopilot"]
            if not ap["running"]: continue

            # 1. SCANNER LOGIC (Buy)
            if ap['cash_balance'] > (ap['total_capital'] * 0.2): 
                scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
                
                # Check if we already have this coin
                if not any(g['coin'] == scan_coin for g in ap['active_grids']):
                    cp = get_current_price(scan_coin)
                    
                    # Simulation: 20% chance to find a setup
                    if random.randint(1, 10) > 8 and cp > 0: 
                        invest_amt = ap['cash_balance'] * 0.2 
                        
                        lower = cp * 0.95
                        upper = cp * 1.05
                        grid_orders = [{"type":"BUY", "price": p} for p in np.linspace(lower, upper, 5)]
                        
                        new_grid = {
                            "coin": scan_coin, "entry": cp, "lower": lower, "upper": upper,
                            "qty": invest_amt/cp, "invest": invest_amt, "grids": 5, "tp": 2.0, "sl": 3.0,
                            "orders": grid_orders
                        }
                        
                        ap['active_grids'].append(new_grid)
                        ap['cash_balance'] -= invest_amt
                        
                        log_msg = f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🚀 AI Deployed Grid: {scan_coin}"
                        ap['logs'].insert(0, log_msg)
                        
            # 2. MONITOR LOGIC (Sell/Stop Loss)
            # We iterate backwards to allow deleting items safely
            for i in range(len(ap['active_grids']) - 1, -1, -1):
                g = ap['active_grids'][i]
                cp = get_current_price(g['coin'])
                
                # Calculate PnL %
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                pnl_pct = (pnl / g['invest']) * 100
                
                # TP (2%) or SL (-3%)
                if pnl_pct >= g['tp'] or pnl_pct <= -g['sl']:
                    reason = "TP Hit ✅" if pnl_pct > 0 else "SL Hit ❌"
                    
                    # Close Trade
                    ap['cash_balance'] += curr_val
                    ap['history'].append({
                        "date": dt.datetime.now(IST), 
                        "pnl": pnl, 
                        "invested": g['invest'],
                        "return_pct": pnl_pct
                    })
                    
                    log_msg = f"[{dt.datetime.now(IST).strftime('%H:%M')}] {reason}: Closed {g['coin']} (${pnl:.2f})"
                    ap['logs'].insert(0, log_msg)
                    
                    # Send Telegram Alert
                    usd_inr = get_usd_inr_rate()
                    tg_msg = (f"🚨 *Auto-Pilot Alert*\n"
                              f"Action: {reason}\n"
                              f"Asset: {g['coin']}\n"
                              f"💰 PnL: ${pnl:.2f} (₹{pnl*usd_inr:,.2f})\n"
                              f"Return: {pnl_pct:.2f}%")
                    send_telegram_alert(tg_msg)
                    
                    ap['active_grids'].pop(i)

        except Exception as e:
            print(f"Background Thread Error: {e}")
            time.sleep(5)

# --- PAGE: MANUAL BOT ---
def show_manual_bot():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="manual_refresh")
    
    # ... (Keep existing manual bot logic, charts, and tables) ...
    # For brevity, reusing the logic from previous steps. 
    # Just ensure this function renders the Manual Bot UI you already have.
    # Below is the abbreviated version:
    
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Price", f"${curr_price:,.4f}")
    with c2:
        invest = st.number_input("Invest (USDT)", value=100.0)
        if st.button("Start Bot"):
            st.session_state["grid_bot_active"][selected_coin] = {
                "coin": selected_coin, "entry_price": curr_price, "invest": invest,
                "qty": invest/curr_price, "lower": curr_price*0.95, "upper": curr_price*1.05
            }
            st.success("Started!")
            
    # Active Manual Bots Table
    active = st.session_state["grid_bot_active"]
    if active:
        for bid, d in list(active.items()):
            cp = get_current_price(d['coin'])
            val = d['qty'] * cp
            pnl = val - d['invest']
            c1,c2,c3,c4 = st.columns(4)
            c1.write(d['coin'])
            c2.write(f"${val:.2f}")
            c3.write(f"${pnl:.2f}")
            if c4.button("Stop", key=f"stop_{bid}"):
                del st.session_state["grid_bot_active"][bid]
                st.rerun()

# --- PAGE: AUTO PILOT UI ---
def show_autopilot():
    st.title("🚀 AI Auto-Pilot")
    # Refresh UI every 10s to reflect background thread changes
    st_autorefresh(interval=10_000, key="ap_refresh") 
    
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    if not ap["running"]:
        c1, c2 = st.columns(2)
        capital = c1.number_input("Capital (USDT)", value=1000.0)
        if c2.button("🚀 Launch Engine"):
            ap["running"] = True
            ap["total_capital"] = capital
            ap["cash_balance"] = capital
            st.rerun()
    else:
        st.success("✅ Engine Running in Background...")
        
        # Metrics
        curr_val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
        total = ap['cash_balance'] + curr_val
        pnl = total - ap['total_capital']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Value", f"${total:,.2f}")
        m2.metric("Cash", f"${ap['cash_balance']:.2f}")
        m3.metric("PnL (USD)", f"${pnl:.2f}", delta=f"{(pnl/ap['total_capital'])*100:.2f}%")
        m4.metric("PnL (INR)", f"₹{pnl*usd_inr:,.0f}")
        
        st.markdown("---")
        st.subheader("💼 Active Grids")
        for i, g in enumerate(ap['active_grids']):
            cp = get_current_price(g['coin'])
            val = g['qty'] * cp
            gpnl = val - g['invest']
            
            c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1.5, 1, 1])
            c1.write(g['coin'])
            c2.write(f"${g['lower']:.4f}-${g['upper']:.4f}")
            c3.write(f"Inv: ${g['invest']:.0f}")
            c4.markdown(f":{'green' if gpnl>=0 else 'red'}[${gpnl:.2f}]")
            
            if c5.button("Stop", key=f"ap_stop_{i}"):
                ap['cash_balance'] += val
                ap['active_grids'].pop(i)
                st.rerun()
        
        st.markdown("---")
        st.subheader("📜 Live Logs")
        for log in ap['logs'][:5]:
            st.text(log)
            
        if st.button("⏹ Stop Engine"):
            ap["running"] = False
            st.rerun()

# --- PAGE: REPORT ---
def show_report():
    st.title("📑 Crypto Report")
    ap = st.session_state["autopilot"]
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        st.dataframe(df)
    else:
        st.info("No closed trades yet.")
