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

# --- BACKGROUND TRADING LOOP ---
def crypto_trading_loop():
    while True:
        try:
            time.sleep(15)
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
                        grid_orders = [{"type": "BUY", "price": p, "status": "OPEN"} for p in np.linspace(lower, upper, 5)]
                        
                        ap['active_grids'].append({
                            "coin": scan_coin, "entry": cp, "lower": lower, "upper": upper,
                            "qty": invest_amt/cp, "invest": invest_amt, 
                            "grids": 5, "tp": 2.0, "sl": 3.0, "orders": grid_orders
                        })
                        ap['cash_balance'] -= invest_amt
                        ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🚀 Deployed: {scan_coin}")

            # 2. MONITOR
            for i in range(len(ap['active_grids']) - 1, -1, -1):
                g = ap['active_grids'][i]
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                pnl_pct = (pnl / g['invest']) * 100
                
                if pnl_pct >= g['tp'] or pnl_pct <= -g['sl']:
                    ap['cash_balance'] += curr_val
                    ap['history'].append({"date": dt.datetime.now(IST), "pnl": pnl, "invested": g['invest'], "return_pct": pnl_pct})
                    
                    usd_inr = get_usd_inr_rate()
                    reason = "TP Hit ✅" if pnl_pct > 0 else "SL Hit ❌"
                    msg = f"🚨 *Auto-Pilot Closed*\nAsset: {g['coin']}\n{reason}\nPnL: ${pnl:.2f} (₹{pnl*usd_inr:.0f})"
                    send_telegram_alert(msg)
                    ap['logs'].insert(0, f"{reason}: {g['coin']}")
                    
                    ap['active_grids'].pop(i)
        except: time.sleep(5)

# --- PAGE: MANUAL BOT ---
def show_crypto_manual_bot_page():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="manual_refresh")
    usd_inr = get_usd_inr_rate()

    # 1. Config Panel
    st.subheader("⚙️ Configure Bot")
    c1, c2 = st.columns([1, 2])
    with c1:
        # The CSS in utils.py forces this to be white/black
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Current Price", f"${curr_price:,.4f}")
        st.caption(f"≈ ₹{curr_price * usd_inr:,.2f}")
    
    with c2:
        col_a, col_b = st.columns(2)
        invest = col_a.number_input("Invest (USDT)", value=100.0, step=10.0)
        grids = col_b.number_input("Grids", value=5, min_value=2)
        
        col_c, col_d = st.columns(2)
        lower_p = col_c.number_input("Lower Price", value=curr_price*0.95, format="%.4f")
        upper_p = col_d.number_input("Upper Price", value=curr_price*1.05, format="%.4f")
        
        # ADDED: TP/SL INPUTS
        col_e, col_f = st.columns(2)
        tp_pct = col_e.number_input("Take Profit (%)", value=2.0, step=0.1)
        sl_pct = col_f.number_input("Stop Loss (%)", value=3.0, step=0.1)
        
        if st.button("▶️ Start Manual Bot", type="primary"):
            st.session_state["grid_bot_active"][selected_coin] = {
                "coin": selected_coin, "entry_price": curr_price, "invest": invest,
                "qty": invest/curr_price, "lower": lower_p, "upper": upper_p,
                "grids": grids, "tp": tp_pct, "sl": sl_pct
            }
            st.success("Bot Started!")

    # 2. Chart (Moved Up)
    st.markdown("---")
    st.subheader(f"📉 {selected_coin} Chart")
    df = get_safe_crypto_data(selected_coin)
    if df is not None:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(height=400, plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)
    
    # 3. Active Bots Table
    st.markdown("---")
    st.subheader("📍 Active Manual Bots")
    active = st.session_state["grid_bot_active"]
    if active:
        # Header
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.write("**Coin**"); c2.write("**Entry**"); c3.write("**Inv**")
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
                send_telegram_alert(f"🚨 Manual Bot Stopped: {d['coin']} PnL: ${pnl:.2f} (₹{pnl_inr:.0f})")
                del st.session_state["grid_bot_active"][bid]
                st.rerun()
    else:
        st.info("No active manual bots.")

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
        if c3.button("🚀 Launch Engine", type="primary"):
            ap["running"] = True
            ap["currency"] = "USDT" if curr_mode == "USDT" else "INR"
            if ap["currency"] == "INR":
                ap["total_capital"] = cap_input / usd_inr
                ap["cash_balance"] = cap_input / usd_inr
            else:
                ap["total_capital"] = cap_input
                ap["cash_balance"] = cap_input
            ap["active_grids"] = []
            ap["logs"].append(f"Engine Started ({curr_mode})")
            st.rerun()
    else:
        st.success(f"✅ AI Engine Active (Updated: {dt.datetime.now(IST).strftime('%H:%M:%S')})")
        
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
            # EXPANDED COLUMNS FOR COMPLETENESS
            c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8])
            c1.markdown("**Asset**"); c2.markdown("**Entry**"); c3.markdown("**Qty**")
            c4.markdown("**Current**"); c5.markdown("**Inv**"); c6.markdown("**Val**")
            c7.markdown("**PnL**"); c8.markdown("**Act**")
            
            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                val = g['qty'] * cp
                gpnl = val - g['invest']
                
                c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 0.8])
                c1.write(g['coin'])
                c2.write(f"${g['entry']:.4f}")
                c3.write(f"{g['qty']:.4f}")
                c4.write(f"${cp:.4f}")
                c5.write(f"${g['invest']:.2f}")
                c6.write(f"${val:.2f}")
                c7.markdown(f":{'green' if gpnl>=0 else 'red'}[${gpnl:.2f}]")
                
                if c8.button("Stop", key=f"ap_stop_{i}"):
                    ap['cash_balance'] += val
                    ap['history'].append({"date": dt.datetime.now(IST), "pnl": gpnl, "invested": g['invest']})
                    send_telegram_alert(f"🚨 Auto-Pilot Closed: {g['coin']} PnL: ${gpnl:.2f}")
                    ap['active_grids'].pop(i)
                    st.rerun()
            
            # LIVE ORDERS
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
    
    invested = sum([g.get('invest', 0) for g in ap['active_grids']])
    curr_val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
    pnl = curr_val - invested
    
    st.subheader("🔴 Live Portfolio")
    c1, c2, c3 = st.columns(3)
    c1.metric("Invested (INR)", f"₹{invested*usd_inr:,.0f}")
    c2.metric("Value (INR)", f"₹{curr_val*usd_inr:,.0f}")
    c3.metric("PnL (INR)", f"₹{pnl*usd_inr:,.0f}")
    
    st.markdown("---")
    st.subheader("🏁 Closed Trades")
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        df['date'] = pd.to_datetime(df['date'])
        tot = df['pnl'].sum()
        st.metric("Total Realized PnL", f"${tot:.2f} (₹{tot*usd_inr:,.0f})")
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
    else:
        st.info("No closed trades.")
