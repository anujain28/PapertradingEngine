import streamlit as st
import pandas as pd
import requests
import datetime as dt
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
from utils import IST, send_telegram_alert

AIROBOTS_URL = "https://airobots.streamlit.app/"
MARKET_START = dt.time(9, 15)
MARKET_END = dt.time(15, 30)

def init_stock_state():
    if "stocks_engine" not in st.session_state:
        st.session_state["stocks_engine"] = {
            "running": False, "capital": 100000.0, "cash": 100000.0,
            "portfolio": [], "history": [], "logs": [], "top_10": [], "btst_picks": []
        }

@st.cache_data(ttl=300)
def fetch_stocks_data():
    try:
        tables = pd.read_html(requests.get(AIROBOTS_URL).text)
        if tables:
            df = tables[0]; df.columns = [c.lower() for c in df.columns]
            return df.head(10).to_dict('records'), []
    except: return [], []
    return [], []

def get_live_price(ticker):
    try: return yf.Ticker(f"{ticker}.NS").history(period="1d")["Close"].iloc[-1]
    except: return 0.0

def run_trading_cycle():
    se = st.session_state["stocks_engine"]
    if not se["running"]: return
    top_10, _ = fetch_stocks_data()
    if top_10: se["top_10"] = top_10
    
    if se["cash"] > se["capital"] * 0.1:
        for stock in se["top_10"][:2]:
            sym = list(stock.values())[0] 
            if not any(p['symbol'] == sym for p in se['portfolio']):
                price = get_live_price(sym)
                if price > 0:
                    qty = int((se["capital"]*0.15)/price)
                    se["portfolio"].append({"symbol": sym, "entry": price, "qty": qty})
                    se["cash"] -= qty * price
                    se["logs"].insert(0, f"Bought {sym} @ {price}")

    for i, p in enumerate(se["portfolio"]):
        curr = get_live_price(p["symbol"])
        if curr > 0:
            pnl = (curr - p["entry"]) * p["qty"]
            # Simple simulation logic for demo
            if pnl > 500 or pnl < -500:
                se["cash"] += curr * p["qty"]
                se["history"].append({"symbol": p["symbol"], "pnl": pnl})
                send_telegram_alert(f"Stock Closed: {p['symbol']} PnL: {pnl}")
                se["portfolio"].pop(i)

def show_bomb_stocks_page():
    st.title("💣 Bomb Stocks")
    st_autorefresh(interval=300_000, key="stock_refresh")
    run_trading_cycle()
    se = st.session_state["stocks_engine"]
    
    c1, c2, c3 = st.columns([1, 1, 2])
    if c1.button("🚀 Start"): se["running"] = True
    if c2.button("⏹ Stop"): se["running"] = False
    with c3: st.markdown(f"**Status:** {'RUNNING' if se['running'] else 'STOPPED'} | **Cash:** ₹{se['cash']:,.0f}")

    st.markdown("---")
    st.subheader("🎯 Top 10 High Velocity")
    if se["top_10"]: st.dataframe(pd.DataFrame(se["top_10"]), use_container_width=True)
    else: st.info("Waiting for AI...")

    st.markdown("---")
    st.subheader("💼 Active Portfolio")
    if se["portfolio"]:
        for i, p in enumerate(se["portfolio"]):
            curr = get_live_price(p["symbol"])
            pnl = (curr - p["entry"]) * p["qty"]
            c1, c2, c3, c4 = st.columns(4)
            c1.write(p["symbol"]); c2.write(f"Entry: {p['entry']}"); c3.write(f"PnL: {pnl:.2f}")
            if c4.button("Sell", key=f"s_{i}"):
                se["cash"] += curr * p["qty"]
                se["portfolio"].pop(i)
                st.rerun()
    else: st.write("No active trades.")

    with st.expander("Logs", expanded=True):
        for log in se["logs"][:5]: st.text(log)

def run_stocks_app():
    init_stock_state()
    show_bomb_stocks_page()
