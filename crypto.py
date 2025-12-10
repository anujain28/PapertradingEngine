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
    else:
        # Check for missing keys in existing state
        defaults = {
            "running": False, "mode": "PAPER", "currency": "USDT",
            "total_capital": 0.0, "cash_balance": 0.0,
            "active_grids": [], "logs": [], "history": []
        }
        for k, v in defaults.items():
            if k not in st.session_state["autopilot"]:
                st.session_state["autopilot"][k] = v

# --- BACKGROUND THREAD LOGIC ---
def crypto_trading_loop():
    """Background thread that runs scanning and trading logic."""
    while True:
        try:
            time.sleep(15) # Scan every 15s
            
            if "autopilot" not in st.session_state: continue
            ap = st.session_state["autopilot"]
            if not ap["running"]: continue

            # 1. SCANNER
            if ap['cash_balance'] > (ap['total_capital'] * 0.2):
                scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
                if not any(g['coin'] == scan_coin for g in ap['active_grids']):
                    cp = get_current_price(scan_coin)
                    
                    if random.randint(1, 10) > 8 and cp > 0:
                        invest_amt = ap['cash_balance'] * 0.2
                        lower = cp * 0.95; upper = cp * 1.05
                        
                        # Generate orders for display
                        grid_orders = [{"type": "BUY", "price": p, "status": "OPEN"} for p in np.linspace(lower, upper, 5)]
                        
                        ap['active_grids'].append({
                            "coin": scan_coin, "entry": cp, "lower": lower, "upper": upper,
                            "qty": invest_amt/cp, "invest": invest_amt, 
                            "grids": 5, "tp": 2.0, "sl": 3.0, "orders": grid_orders
                        })
                        ap['cash_balance'] -= invest_amt
                        
                        log = f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🚀 Deployed Grid: {scan_coin}"
                        ap['logs'].insert(0, log)

            # 2. MONITOR
            for i in range(len(ap['active_grids']) - 1, -1, -1):
                g = ap['active_grids'][i]
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                pnl_pct = (pnl / g['invest']) * 100
                
                # Check TP/SL
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
                           f"PnL: ${pnl:.2f} (₹{pnl*usd_inr:.0f})\nReturn: {pnl_pct:.2f}%")
                    send_telegram_alert(msg)
                    
                    ap['active_grids'].pop(i)

        except Exception as e:
            print(f"Thread Error: {e}")
            time.sleep(5)

# --- PAGE: MANUAL BOT ---
def show_crypto_manual_bot_page():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="manual_refresh")
    usd_inr = get_usd_inr_rate()

    st.subheader("🔎 Market Analysis")
    analysis_data = []
    for coin in CRYPTO_SYMBOLS_USD:
        hist = get_safe_crypto_data(coin, period="5d")
        if hist is not None:
            curr = hist['Close'].iloc[-1]
            prev = hist['Close'].iloc[-2]
            change = ((curr - prev)/prev)*100
            analysis_data.append({
                "Coin": coin, "Price": f"${curr:,.2f}", "Change": f"{change:+.2f}%"
            })
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    st.markdown("---")

    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Price", f"${curr_price:,.4f}")
    
    with c2:
        col_a, col_b = st.columns(2)
        invest = st.number_input("Invest (USDT)", value=100.0)
        grids = st.number_input("Grids", value=5)
        
        if st.button("▶️ Start Manual Bot"):
            st.session_state["grid_bot_active"][selected_coin] = {
                "coin": selected_coin, "entry_price": curr_price, "invest": invest,
                "qty": invest/curr_price, "lower": curr_price*0.95, "upper": curr_price*1.05,
                "grids": grids, "tp": 2.0
            }
            st.success("Started!")

    st.markdown("---")
    st.subheader("📍 Active Manual Bots")
    active = st.session_state["grid_bot_active"]
    if active:
        for bid, d in list(active.items()):
            cp = get_current_price(d['coin'])
            val = d['qty'] * cp
            pnl = val - d['invest']
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.write(d['coin'])
            c2.write(f"Inv: ${d['invest']:.2f}")
            c3.write(f"Val: ${val:.2f}")
            c4.markdown(f":{'green' if pnl>=0 else 'red'}[${pnl:.2f}]")
            
            if c5.button("Stop", key=f"stop_{bid}"):
                pnl_inr = pnl * usd_inr
                msg = f"🚨 *Manual Bot Stopped*\nAsset: {d['coin']}\nPnL: ${pnl:.2f} (₹{pnl_inr:.0f})"
                send_telegram_alert(msg)
                del st.session_state["grid_bot_active"][bid]
                st.rerun()
    else:
        st.info("No active bots.")

    # Chart
    st.markdown("---")
    st.subheader(f"📉 {selected_coin} Chart")
    df = get_safe_crypto_data(selected_coin)
    if df is not None:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(height=500, plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE: AUTO PILOT ---
def show_ai_autopilot_page():
    st.title("🚀 Crypto Auto Pilot")
    st_autorefresh(interval=15_000, key="ap_refresh")
    
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    if not ap["running"]:
        c1, c2, c3 = st.columns(3)
        curr_mode = c1.radio("Currency", ["USDT", "INR"])
        cap_input = c2.number_input("Capital", value=1000.0)
        
        if c3.button("🚀 Launch"):
            ap["running"] = True
            ap["currency"] = "USDT" if curr_mode == "USDT" else "INR"
            
            # Normalize to USDT for logic
            if ap["currency"] == "INR":
                ap["total_capital"] = cap_input / usd_inr
                ap["cash_balance"] = cap_input / usd_inr
            else:
                ap["total_capital"] = cap_input
                ap["cash_balance"] = cap_input
            
            ap["active_grids"] = []
            ap["logs"].append(f"Engine Started in {curr_mode}")
            st.rerun()
    else:
        st.success(f"✅ AI Engine Active (Updated: {dt.datetime.now(IST).strftime('%H:%M:%S')})")
        
        # Metrics
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
        st.subheader("💼 Active Grids")
        if ap['active_grids']:
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.2, 1.2, 1.2, 1.2, 0.8])
            c1.write("**Asset**"); c2.write("**Range**"); c3.write("**Config**")
            c4.write("**Inv**"); c5.write("**Val**"); c6.write("**PnL**"); c7.write("**Act**")
            
            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                val = g['qty'] * cp
                gpnl = val - g['invest']
                
                c1, c2, c3, c4, c5, c6, c7 = st.columns([1, 1.5, 1.2, 1.2, 1.2, 1.2, 0.8])
                c1.write(g['coin'])
                c2.write(f"${g['lower']:.4f}-${g['upper']:.4f}")
                c3.write(f"{g.get('grids',5)} G | TP {g.get('tp',2.0)}%")
                c4.write(f"${g['invest']:.2f}")
                c5.write(f"${val:.2f}")
                c6.markdown(f":{'green' if gpnl>=0 else 'red'}[${gpnl:.2f}]")
                
                if c7.button("Stop", key=f"ap_stop_{i}"):
                    ap['cash_balance'] += val
                    ap['history'].append({"date": dt.datetime.now(IST), "pnl": gpnl, "invested": g['invest']})
                    
                    msg = f"🚨 *Auto-Pilot Closed*\nAsset: {g['coin']}\nPnL: ₹{gpnl*usd_inr:.2f}"
                    send_telegram_alert(msg)
                    
                    ap['active_grids'].pop(i)
                    st.rerun()
            
            st.markdown("---")
            st.markdown("### 📋 Grid Orders")
            for g in ap['active_grids']:
                with st.expander(f"Orders for {g['coin']}"):
                    st.table(pd.DataFrame(g.get('orders', [])))
        
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
    st.title("📑 Crypto Report")
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
