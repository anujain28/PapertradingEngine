import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import time
import random
import yfinance as yf
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh
# Safe Import
from utils import IST, send_telegram_alert, get_usd_inr_rate

CRYPTO_SYMBOLS_USD = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]

# --- HELPER ---
def get_current_price(symbol):
    try: return yf.Ticker(symbol).history(period="1d")["Close"].iloc[-1]
    except: return 0.0

@st.cache_data(ttl=300)
def get_safe_crypto_data(symbol, period="1mo"):
    try: return yf.Ticker(symbol).history(period=period)
    except: return None

# --- STATE ---
def init_crypto_state():
    if "grid_bot_active" not in st.session_state:
        st.session_state["grid_bot_active"] = {} 
    
    if "autopilot" not in st.session_state:
        st.session_state["autopilot"] = {
            "running": False, "mode": "PAPER", "currency": "USDT",
            "total_capital": 0.0, "cash_balance": 0.0,
            "active_grids": [], "logs": [], "history": []
        }

# --- BACKGROUND THREAD ---
def crypto_trading_loop():
    while True:
        try:
            time.sleep(15)
            if "autopilot" not in st.session_state: continue
            ap = st.session_state["autopilot"]
            if not ap["running"]: continue

            # SCANNER
            if ap['cash_balance'] > (ap['total_capital'] * 0.2):
                scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
                if not any(g['coin'] == scan_coin for g in ap['active_grids']):
                    cp = get_current_price(scan_coin)
                    if random.randint(1, 10) > 8 and cp > 0:
                        invest_amt = ap['cash_balance'] * 0.2
                        lower = cp * 0.95; upper = cp * 1.05
                        orders = [{"type": "BUY", "price": p, "status": "OPEN"} for p in np.linspace(lower, upper, 5)]
                        
                        ap['active_grids'].append({
                            "coin": scan_coin, "entry": cp, "lower": lower, "upper": upper,
                            "qty": invest_amt/cp, "invest": invest_amt, 
                            "grids": 5, "tp": 2.0, "sl": 3.0, "orders": orders
                        })
                        ap['cash_balance'] -= invest_amt
                        ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🚀 Deployed: {scan_coin}")

            # MONITOR
            for i in range(len(ap['active_grids']) - 1, -1, -1):
                g = ap['active_grids'][i]
                cp = get_current_price(g['coin'])
                curr_val = g['qty'] * cp
                pnl = curr_val - g['invest']
                pnl_pct = (pnl / g['invest']) * 100
                
                if pnl_pct >= g['tp'] or pnl_pct <= -g['sl']:
                    ap['cash_balance'] += curr_val
                    ap['history'].append({
                        "date": dt.datetime.now(IST), "pnl": pnl, "invested": g['invest'], "return_pct": pnl_pct
                    })
                    usd_inr = get_usd_inr_rate()
                    send_telegram_alert(f"🚨 Auto-Pilot Closed: {g['coin']} PnL: ${pnl:.2f} (₹{pnl*usd_inr:.0f})")
                    ap['logs'].insert(0, f"Closed {g['coin']}: ${pnl:.2f}")
                    ap['active_grids'].pop(i)
        except: time.sleep(5)

# --- PAGE 1: MANUAL BOT ---
def show_crypto_manual_bot_page():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="manual_refresh")
    usd_inr = get_usd_inr_rate()

    # 1. Analysis
    st.subheader("🔎 Market Analysis")
    data = []
    for c in CRYPTO_SYMBOLS_USD:
        p = get_current_price(c)
        data.append({"Coin": c, "Price": f"${p:.2f}"})
    st.dataframe(pd.DataFrame(data), use_container_width=True)
    st.markdown("---")

    # 2. Config
    st.subheader("⚙️ Configure Bot")
    c1, c2 = st.columns([1, 2])
    with c1:
        sel_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr = get_current_price(sel_coin)
        st.metric("Price", f"${curr:.4f}")
        st.caption(f"₹{curr*usd_inr:.2f}")
    
    with c2:
        col1, col2 = st.columns(2)
        inv = col1.number_input("Invest (USDT)", 100.0, step=10.0)
        grids = col2.number_input("Grids", 5)
        
        col3, col4 = st.columns(2)
        lp = col3.number_input("Lower", value=curr*0.95, format="%.4f")
        up = col4.number_input("Upper", value=curr*1.05, format="%.4f")
        
        col5, col6 = st.columns(2)
        tp = col5.number_input("TP (%)", 2.0)
        sl = col6.number_input("SL (%)", 3.0)
        
        if st.button("▶️ Start Bot", type="primary"):
            st.session_state["grid_bot_active"][sel_coin] = {
                "coin": sel_coin, "entry_price": curr, "invest": inv,
                "qty": inv/curr, "lower": lp, "upper": up, "grids": grids, "tp": tp, "sl": sl
            }
            st.success("Started!")

    # 3. Chart
    st.markdown("---")
    st.subheader(f"📉 {sel_coin} Chart")
    df = get_safe_crypto_data(sel_coin)
    if df is not None:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(color='black'), xaxis=dict(gridcolor='#eee'), yaxis=dict(gridcolor='#eee'))
        st.plotly_chart(fig, use_container_width=True)

    # 4. Active
    st.markdown("---")
    st.subheader("📍 Active Manual Bots")
    active = st.session_state["grid_bot_active"]
    if active:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.write("Coin"); c2.write("Entry"); c3.write("Inv"); c4.write("Val"); c5.write("PnL"); c6.write("Act")
        for bid, d in list(active.items()):
            cp = get_current_price(d['coin'])
            val = d['qty'] * cp
            pnl = val - d['invest']
            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.write(d['coin'])
            c2.write(f"${d['entry_price']:.4f}")
            c3.write(f"${d['invest']:.2f}")
            c4.write(f"${val:.2f}")
            c5.markdown(f":{'green' if pnl>0 else 'red'}[${pnl:.2f}]")
            if c6.button("Stop", key=f"s_{bid}"):
                del st.session_state["grid_bot_active"][bid]
                st.rerun()
    else: st.info("No active bots.")

# --- PAGE 2: AUTO PILOT ---
def show_ai_autopilot_page():
    st.title("🚀 Crypto Auto Pilot")
    st_autorefresh(interval=15_000, key="ap_refresh")
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    if not ap["running"]:
        c1, c2, c3 = st.columns(3)
        mode = c1.radio("Currency", ["USDT", "INR"])
        cap = c2.number_input("Capital", 1000.0, step=100.0)
        if c3.button("🚀 Launch", type="primary"):
            ap["running"] = True; ap["currency"] = mode
            ap["total_capital"] = cap if mode == "USDT" else cap/usd_inr
            ap["cash_balance"] = ap["total_capital"]
            ap["active_grids"] = []; ap["logs"].append("Started")
            st.rerun()
    else:
        st.success(f"✅ Engine Active (Time: {dt.datetime.now(IST).strftime('%H:%M:%S')})")
        
        # Metrics
        curr_sym = "$" if ap["currency"] == "USDT" else "₹"
        fac = 1.0 if ap["currency"] == "USDT" else usd_inr
        
        inv = sum([g.get('invest', 0) for g in ap['active_grids']])
        val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
        tot = ap['cash_balance'] + val
        pnl = tot - ap['total_capital']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total", f"{curr_sym}{tot*fac:,.2f}")
        m2.metric("Cash", f"{curr_sym}{ap['cash_balance']*fac:,.2f}")
        m3.metric("Invested", f"{curr_sym}{inv*fac:,.2f}")
        m4.metric("PnL", f"{curr_sym}{pnl*fac:,.2f}", delta=f"{(pnl/ap['total_capital'])*100:.2f}%")
        
        st.markdown("---")
        st.subheader("💼 Active Grids")
        if ap['active_grids']:
            c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([1,1.2,1.2,1.2,1.2,1.2,1.2,0.8])
            c1.write("**Asset**"); c2.write("**Range**"); c3.write("**Config**"); c4.write("**Entry**")
            c5.write("**Inv**"); c6.write("**Val**"); c7.write("**PnL**"); c8.write("**Act**")
            
            for i, g in enumerate(ap['active_grids']):
                cp = get_current_price(g['coin'])
                val = g['qty'] * cp
                gpnl = val - g['invest']
                
                c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([1,1.2,1.2,1.2,1.2,1.2,1.2,0.8])
                c1.write(g['coin'])
                c2.write(f"${g['lower']:.4f}-${g['upper']:.4f}")
                c3.write(f"{g.get('grids',5)}G | TP {g.get('tp',2.0)}%")
                c4.write(f"${g['entry']:.4f}")
                c5.write(f"${g['invest']:.2f}")
                c6.write(f"${val:.2f}")
                c7.markdown(f":{'green' if gpnl>=0 else 'red'}[${gpnl:.2f}]")
                if c8.button("Stop", key=f"ap_{i}"):
                    ap['cash_balance'] += val
                    ap['history'].append({"date": dt.datetime.now(IST), "pnl": gpnl, "invested": g['invest']})
                    ap['active_grids'].pop(i)
                    st.rerun()
                
                with st.expander(f"Orders for {g['coin']}"):
                    if 'orders' in g: st.table(pd.DataFrame(g['orders']))
        else: st.info("Scanning...")
        
        if st.button("⏹ Stop"): ap["running"] = False; st.rerun()

# --- PAGE 3: REPORT ---
def show_crypto_report_page():
    st.title("📑 Report")
    st_autorefresh(interval=30000)
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    inv = sum([g.get('invest', 0) for g in ap['active_grids']])
    val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
    pnl = val - inv
    
    st.subheader("🔴 Live Portfolio")
    c1, c2, c3 = st.columns(3)
    c1.metric("Inv (INR)", f"₹{inv*usd_inr:,.0f}")
    c2.metric("Val (INR)", f"₹{val*usd_inr:,.0f}")
    c3.metric("PnL (INR)", f"₹{pnl*usd_inr:,.0f}")
    
    st.markdown("---")
    st.subheader("🏁 History")
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        df['date'] = pd.to_datetime(df['date'])
        tot = df['pnl'].sum()
        st.metric("Total PnL", f"${tot:.2f} (₹{tot*usd_inr:.0f})")
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)
    else: st.info("No history.")
