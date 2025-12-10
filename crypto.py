import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import random
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
from utils import IST, send_telegram_alert, get_usd_inr_rate

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

# --- STATE INITIALIZATION ---
def init_crypto_state():
    if "grid_bot_active" not in st.session_state:
        st.session_state["grid_bot_active"] = {} 
    
    if "autopilot" not in st.session_state:
        st.session_state["autopilot"] = {
            "running": False, "mode": "PAPER", "currency": "USDT",
            "total_capital": 0.0, "cash_balance": 0.0,
            "active_grids": [], "logs": [], "history": []
        }
    else:
        # Restore missing keys if any
        defaults = {
            "running": False, "mode": "PAPER", "currency": "USDT",
            "total_capital": 0.0, "cash_balance": 0.0,
            "active_grids": [], "logs": [], "history": []
        }
        for k, v in defaults.items():
            if k not in st.session_state["autopilot"]:
                st.session_state["autopilot"][k] = v

# --- BACKGROUND TRADING LOOP ---
def crypto_trading_loop():
    """Background thread: Scans for buys and monitors sells for Auto-Pilot."""
    while True:
        try:
            time.sleep(15) # Pulse every 15s
            
            if "autopilot" not in st.session_state: continue
            ap = st.session_state["autopilot"]
            if not ap["running"]: continue

            # 1. SCANNER LOGIC
            if ap['cash_balance'] > (ap['total_capital'] * 0.2):
                scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
                # Ensure we don't double trade same coin
                if not any(g['coin'] == scan_coin for g in ap['active_grids']):
                    cp = get_current_price(scan_coin)
                    
                    # Simulation Chance (Replace with RSI logic if needed)
                    if random.randint(1, 10) > 8 and cp > 0:
                        invest_amt = ap['cash_balance'] * 0.2
                        lower = cp * 0.95; upper = cp * 1.05
                        
                        # Generate Grid Levels for UI
                        grid_orders = [{"type": "BUY", "price": p, "status": "OPEN"} for p in np.linspace(lower, upper, 5)]
                        
                        ap['active_grids'].append({
                            "coin": scan_coin, "entry": cp, "lower": lower, "upper": upper,
                            "qty": invest_amt/cp, "invest": invest_amt, 
                            "grids": 5, "tp": 2.0, "sl": 3.0, "orders": grid_orders
                        })
                        ap['cash_balance'] -= invest_amt
                        ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🚀 Deployed Grid: {scan_coin}")

            # 2. MONITOR LOGIC
            for i in range(len(ap['active_grids']) - 1, -1, -1):
                g = ap['active_grids'][i]
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                pnl_pct = (pnl / g['invest']) * 100
                
                # Check Exits
                if pnl_pct >= g['tp'] or pnl_pct <= -g['sl']:
                    ap['cash_balance'] += curr_val
                    ap['history'].append({
                        "date": dt.datetime.now(IST), "pnl": pnl, 
                        "invested": g['invest'], "return_pct": pnl_pct
                    })
                    
                    reason = "TP Hit ✅" if pnl_pct > 0 else "SL Hit ❌"
                    log = f"[{dt.datetime.now(IST).strftime('%H:%M')}] {reason}: Closed {g['coin']} (${pnl:.2f})"
                    ap['logs'].insert(0, log)
                    
                    # Telegram
                    usd_inr = get_usd_inr_rate()
                    msg = (f"🚨 *Auto-Pilot Closed*\nAsset: {g['coin']}\n"
                           f"PnL: ${pnl:.2f} (₹{pnl*usd_inr:,.0f})\nReturn: {pnl_pct:.2f}%")
                    send_telegram_alert(msg)
                    
                    ap['active_grids'].pop(i)

        except Exception:
            time.sleep(5)

# --- UI: MANUAL BOT ---
def show_crypto_manual_bot_page():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="manual_refresh")
    usd_inr = get_usd_inr_rate()

    # 1. Config
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Price", f"${curr_price:,.4f}")
        st.caption(f"≈ ₹{curr_price * usd_inr:,.2f}")
    
    with c2:
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower", value=curr_price*0.95, format="%.4f")
        upper_p = col_b.number_input("Upper", value=curr_price*1.05, format="%.4f")
        invest = st.number_input("Invest (USDT)", value=100.0)
        
        if st.button("▶️ Start Manual Bot"):
            st.session_state["grid_bot_active"][selected_coin] = {
                "coin": selected_coin, "entry_price": curr_price, "invest": invest,
                "qty": invest/curr_price, "lower": lower_p, "upper": upper_p,
                "grids": 5, "tp": 2.0
            }
            st.success("Bot Started!")

    # 2. Active Bots
    st.markdown("---")
    st.subheader("📍 Active Manual Bots")
    active = st.session_state["grid_bot_active"]
    if active:
        # Header
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.write("**Coin**"); c2.write("**Entry**"); c3.write("**Invested**")
        c4.write("**Value**"); c5.write("**PnL**"); c6.write("**Action**")
        
        for bid, d in list(active.items()):
            cp = get_current_price(d['coin'])
            val = d['qty'] * cp
            pnl = val - d['invest']
            
            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.write(d['coin'])
            c2.write(f"${d['entry_price']:.4f}")
            c3.write(f"${d['invest']:.2f}")
            c4.write(f"${val:.2f}")
            c5.markdown(f":{'green' if pnl>=0 else 'red'}[${pnl:.2f}]")
            
            if c6.button("Stop", key=f"stop_{bid}"):
                pnl_inr = pnl * usd_inr
                msg = f"🚨 *Manual Bot Stopped*\nAsset: {d['coin']}\nPnL: ${pnl:.2f} (₹{pnl_inr:,.0f})"
                send_telegram_alert(msg)
                del st.session_state["grid_bot_active"][bid]
                st.rerun()
    else:
        st.info("No active manual bots.")

# --- UI: AUTO PILOT ---
def show_ai_autopilot_page():
    st.title("🚀 Crypto Auto Pilot")
    st_autorefresh(interval=15_000, key="ap_refresh")
    
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    # 1. Config Section
    if not ap["running"]:
        c1, c2, c3 = st.columns(3)
        curr_mode = c1.radio("Currency", ["USDT", "INR"])
        cap_input = c2.number_input("Capital", value=1000.0)
        
        if c3.button("🚀 Launch Engine"):
            ap["running"] = True
            ap["currency"] = "USDT" if curr_mode == "USDT" else "INR"
            
            if ap["currency"] == "INR":
                ap["total_capital"] = cap_input / usd_inr
                ap["cash_balance"] = cap_input / usd_inr
            else:
                ap["total_capital"] = cap_input
                ap["cash_balance"] = cap_input
                
            ap["active_grids"] = []
            ap["logs"].append("Engine Started")
            st.rerun()
    else:
        # 2. Live Dashboard
        st.success(f"✅ AI Engine Active (Last Update: {dt.datetime.now(IST).strftime('%H:%M:%S')})")
        
        curr_sym = "$" if ap["currency"] == "USDT" else "₹"
        factor = 1.0 if ap["currency"] == "USDT" else usd_inr
        
        invested = sum([g.get('invest', 0) for g in ap['active_grids']])
        curr_val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
        total = ap['cash_balance'] + curr_val
        pnl = total - ap['total_capital']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Value", f"{curr_sym}{total*factor:,.2f}")
        m2.metric("Cash", f"{curr_sym}{ap['cash_balance']*factor:,.2f}")
        m3.metric("Invested", f"{curr_sym}{invested*factor:,.2f}")
        m4.metric("Net PnL", f"{curr_sym}{pnl*factor:,.2f}", delta=f"{(pnl/ap['total_capital'])*100:.2f}%")
        
        st.markdown("---")
        
        # 3. Active Grids Table
        st.subheader("💼 Active Grids")
        if ap['active_grids']:
            # Headers
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.5, 1, 1, 1, 0.8])
            c1.markdown("**Asset**"); c2.markdown("**Range**"); c3.markdown("**Config**")
            c4.markdown("**Inv**"); c5.markdown("**Val**"); c6.markdown("**PnL**"); c7.markdown("**Act**")
            
            sum_inv = 0; sum_val = 0; sum_pnl = 0
            
            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                val = g['qty'] * cp
                gpnl = val - g['invest']
                sum_inv += g['invest']; sum_val += val; sum_pnl += gpnl
                
                c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.5, 1, 1, 1, 0.8])
                c1.write(g['coin'])
                c2.write(f"${g['lower']:.4f}-${g['upper']:.4f}")
                c3.write(f"{g.get('grids',5)} G | TP {g.get('tp',2.0)}%")
                c4.write(f"${g['invest']:.2f}")
                c5.write(f"${val:.2f}")
                c6.markdown(f":{'green' if gpnl>=0 else 'red'}[${gpnl:.2f}]")
                
                if c7.button("Stop", key=f"ap_stop_{i}"):
                    ap['cash_balance'] += val
                    ap['history'].append({"date": dt.datetime.now(IST), "pnl": gpnl, "invested": g['invest']})
                    
                    msg = f"🚨 *Auto-Pilot Closed*\nAsset: {g['coin']}\nPnL: ₹{gpnl*usd_inr:,.2f}"
                    send_telegram_alert(msg)
                    
                    ap['active_grids'].pop(i)
                    st.rerun()
            
            # Totals Footer
            st.markdown("<div style='border-top:1px solid #ddd; padding-top:5px;'></div>", unsafe_allow_html=True)
            f1, f4, f5, f6 = st.columns([4, 1, 1, 1])
            f1.write("**TOTALS (USD)**")
            f4.write(f"**${sum_inv:,.2f}**")
            f5.write(f"**${sum_val:,.2f}**")
            f6.markdown(f":{'green' if sum_pnl>=0 else 'red'}[**${sum_pnl:,.2f}**]")

            st.markdown("---")
            st.markdown("### 📋 Live Grid Orders")
            for g in ap['active_grids']:
                with st.expander(f"Orders for {g['coin']}"):
                    # Regenerate if missing
                    if 'orders' not in g or not g['orders']:
                        g['orders'] = [{"type": "BUY", "price": p} for p in np.linspace(g['lower'], g['upper'], 5)]
                    
                    ord_df = pd.DataFrame(g['orders'])
                    ord_df['price'] = ord_df['price'].apply(lambda x: f"${x:.4f}")
                    st.table(ord_df)
        
        else:
            st.info("Scanning for opportunities...")
            
        st.markdown("---")
        st.subheader("📜 Logs")
        for log in ap['logs'][:5]:
            st.text(log)
            
        if st.button("⏹ Stop Engine"):
            ap["running"] = False
            st.rerun()

# --- PAGE: REPORT ---
def show_crypto_report_page():
    st.title("📑 Crypto PnL Report")
    st_autorefresh(interval=30_000)
    
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    # Live Summary
    invested = sum([g.get('invest', 0) for g in ap['active_grids']])
    curr_val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
    pnl = curr_val - invested
    
    st.subheader("🔴 Live Portfolio")
    c1, c2, c3 = st.columns(3)
    c1.metric("Invested (INR)", f"₹{invested*usd_inr:,.0f}")
    c2.metric("Value (INR)", f"₹{curr_val*usd_inr:,.0f}")
    c3.metric("Unrealized PnL", f"₹{pnl*usd_inr:,.0f}")
    
    st.markdown("---")
    st.subheader("🏁 Closed Trades")
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        df['date'] = pd.to_datetime(df['date'])
        
        tot_pnl = df['pnl'].sum()
        m1, m2 = st.columns(2)
        m1.metric("Realized PnL", f"${tot_pnl:,.2f} (₹{tot_pnl*usd_inr:,.0f})")
        m2.metric("Total Trades", len(df))
        
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
    else:
        st.info("No closed trades.")
