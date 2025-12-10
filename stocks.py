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
            df = tables[0]
            df.columns = [c.lower() for c in df.columns]
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
    
    # Simple Buy Logic for Demo
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

def show_bomb_stocks_page():
    st.title("💣 Bomb Stocks (Auto-Pilot)")
    st_autorefresh(interval=300_000, key="stock_refresh")
    run_trading_cycle()
    se = st.session_state["stocks_engine"]
    
    c1, c2 = st.columns(2)
    if c1.button("Start Engine"): se["running"] = True
    if c2.button("Stop Engine"): se["running"] = False
    
    st.subheader("Active Portfolio")
    for i, p in enumerate(se["portfolio"]):
        curr = get_live_price(p["symbol"])
        pnl = (curr - p["entry"]) * p["qty"]
        c1, c2, c3, c4 = st.columns(4)
        c1.write(p["symbol"]); c2.write(f"PnL: {pnl:.2f}")
        if c4.button("Sell", key=f"s_{i}"):
            se["cash"] += curr * p["qty"]
            se["portfolio"].pop(i)
            send_telegram_alert(f"Stock Sold: {p['symbol']} PnL: {pnl}")
            st.rerun()
            
def run_stocks_app():
    init_stock_state()
    show_bomb_stocks_page()
