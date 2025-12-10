import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import plotly.graph_objects as go
import random
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
    else:
        # Migration check
        if "active_grids" not in st.session_state["autopilot"]:
            st.session_state["autopilot"]["active_grids"] = []

# --- PAGE: MANUAL BOT ---
def show_manual_bot():
    st.title("🤖 AI Crypto Manual Bot")
    st_autorefresh(interval=30_000, key="manual_refresh")
    usd_inr = get_usd_inr_rate()
    
    st.subheader("🔎 Live Market Analysis")
    # ... (Add Analysis Table Logic Here if needed, simplified for brevity) ...
    
    c1, c2 = st.columns([1, 2])
    with c1:
        selected_coin = st.selectbox("Select Coin", CRYPTO_SYMBOLS_USD)
        curr_price = get_current_price(selected_coin)
        st.metric("Price", f"${curr_price:,.4f}")
    
    with c2:
        col_a, col_b = st.columns(2)
        lower_p = col_a.number_input("Lower Price", value=curr_price*0.95, format="%.4f")
        upper_p = col_b.number_input("Upper Price", value=curr_price*1.05, format="%.4f")
        invest = st.number_input("Investment (USDT)", value=100.0)
    
    if st.button("▶️ Start Manual Bot"):
        bot_id = selected_coin
        st.session_state["grid_bot_active"][bot_id] = {
            "coin": selected_coin, "entry_price": curr_price,
            "lower": lower_p, "upper": upper_p, "invest": invest,
            "qty": invest/curr_price, "grids": 5, "tp": 2.0, "sl": 3.0
        }
        st.success("Bot Started!")

    # Active Bots Table
    st.markdown("---")
    st.subheader("📍 Active Manual Bots")
    active_bots = st.session_state["grid_bot_active"]
    
    if active_bots:
        for b_id, data in list(active_bots.items()):
            cp = get_current_price(data['coin'])
            cur_val = data['qty'] * cp
            pnl = cur_val - data['invest']
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.write(data['coin'])
            c2.write(f"Inv: ${data['invest']:.2f}")
            c3.write(f"Val: ${cur_val:.2f}")
            c4.markdown(f":{'green' if pnl>=0 else 'red'}[${pnl:.2f}]")
            
            if c5.button("Stop", key=f"stop_{b_id}"):
                send_telegram_alert(f"🛑 Manual Bot Stopped: {data['coin']} PnL: ${pnl:.2f}")
                del st.session_state["grid_bot_active"][b_id]
                st.rerun()
    else:
        st.info("No active manual bots.")

    # Chart
    st.markdown("---")
    st.subheader(f"📉 {selected_coin} Chart")
    df = get_safe_crypto_data(selected_coin)
    if df is not None:
        fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
        fig.update_layout(height=500, plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE: AUTO PILOT ---
def show_autopilot():
    st.title("🚀 AI Auto-Pilot")
    st_autorefresh(interval=15_000, key="ap_refresh")
    ap = st.session_state["autopilot"]
    usd_inr = get_usd_inr_rate()
    
    if not ap["running"]:
        capital = st.number_input("Capital (USDT)", value=1000.0)
        if st.button("🚀 Launch Engine"):
            ap["running"] = True; ap["total_capital"] = capital; ap["cash_balance"] = capital
            st.rerun()
    else:
        # Scanner Logic
        if ap['cash_balance'] > (ap['total_capital'] * 0.2):
            scan_coin = random.choice(CRYPTO_SYMBOLS_USD)
            if not any(g['coin'] == scan_coin for g in ap['active_grids']):
                cp = get_current_price(scan_coin)
                if random.randint(1, 10) > 8 and cp > 0:
                    invest = ap['cash_balance'] * 0.2
                    lower = cp*0.95; upper = cp*1.05
                    grid_orders = [{"type":"BUY", "price": p} for p in np.linspace(lower, upper, 5)]
                    
                    ap['active_grids'].append({
                        "coin": scan_coin, "entry": cp, "lower": lower, "upper": upper,
                        "qty": invest/cp, "invest": invest, "grids": 5, "tp": 2.0, "sl": 3.0,
                        "orders": grid_orders
                    })
                    ap['cash_balance'] -= invest
                    ap['logs'].insert(0, f"[{dt.datetime.now(IST).strftime('%H:%M')}] 🚀 Deployed: {scan_coin}")

        # Metrics
        curr_val = sum([g['qty']*get_current_price(g['coin']) for g in ap['active_grids']])
        total = ap['cash_balance'] + curr_val
        pnl = total - ap['total_capital']
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Value", f"${total:,.2f}")
        m2.metric("PnL (USD)", f"${pnl:,.2f}")
        m3.metric("PnL (INR)", f"₹{pnl*usd_inr:,.0f}")
        
        # Active Grids
        st.subheader("💼 Active Grids")
        for i, g in enumerate(ap['active_grids']):
            cp = get_current_price(g['coin'])
            val = g['qty'] * cp
            gpnl = val - g['invest']
            
            c1, c2, c3, c4, c5 = st.columns([1, 1.5, 1.5, 1, 1])
            c1.write(g['coin'])
            c2.write(f"${g['lower']:.4f}-${g['upper']:.4f}")
            c3.write(f"Config: {g['grids']} Grids")
            c4.markdown(f":{'green' if gpnl>=0 else 'red'}[${gpnl:.2f}]")
            
            if c5.button("Stop", key=f"ap_stop_{i}"):
                ap['cash_balance'] += val
                ap['history'].append({"date": dt.datetime.now(IST), "pnl": gpnl, "invested": g['invest']})
                
                # Telegram
                msg = f"🚨 *Auto-Pilot Closed*\nCoin: {g['coin']}\nPnL: ₹{gpnl*usd_inr:.2f}"
                send_telegram_alert(msg)
                
                ap['active_grids'].pop(i)
                st.rerun()
            
            with st.expander(f"View Orders for {g['coin']}"):
                st.table(pd.DataFrame(g.get('orders', [])))

        if st.button("⏹ Stop Engine"):
            ap["running"] = False; st.rerun()

# --- PAGE: REPORT ---
def show_report():
    st.title("📑 Crypto Report")
    ap = st.session_state["autopilot"]
    if ap["history"]:
        df = pd.DataFrame(ap["history"])
        st.dataframe(df)
    else:
        st.info("No data yet.")
