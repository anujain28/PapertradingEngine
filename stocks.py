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
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(AIROBOTS_URL, headers=headers, timeout=15)
        if response.status_code != 200: return None, None
        
        tables = pd.read_html(response.text)
        if not tables: return None, None
        
        velocity_list = []; btst_list = []
        time_factors = [1, 2, 7, 30] 
        
        for i, df in enumerate(tables):
            df.columns = [str(c).lower().strip() for c in df.columns]
            symbol_col = next((c for c in df.columns if any(k in c for k in ['stock', 'symbol'])), None)
            profit_col = next((c for c in df.columns if any(k in c for k in ['profit', 'return'])), None)
            
            if symbol_col and profit_col:
                category = ["Intraday", "BTST", "Weekly", "Monthly"][i] if i < 4 else "Long Term"
                days = time_factors[i] if i < len(time_factors) else 30
                
                for _, row in df.iterrows():
                    try:
                        sym = str(row[symbol_col]).upper().replace(".NS", "").strip()
                        raw = str(row[profit_col]).replace("%", "").strip()
                        prof = float(raw) if raw.replace('.', '', 1).isdigit() else 0.0
                        
                        if prof > 0 and prof < 100:
                            item = {"Symbol": sym, "Type": category, "Exp. Profit": prof, "Velocity": prof/days}
                            velocity_list.append(item)
                            if category == "BTST": btst_list.append(item)
                    except: continue
                    
        top_10 = pd.DataFrame(velocity_list).sort_values(by="Velocity", ascending=False).head(10).to_dict('records') if velocity_list else []
        btst = pd.DataFrame(btst_list).sort_values(by="Exp. Profit", ascending=False).head(10).to_dict('records') if btst_list else []
        return top_10, btst
    except: return None, None

def get_live_price(ticker):
    try:
        return yf.Ticker(f"{ticker}.NS").history(period="1d")["Close"].iloc[-1]
    except: return 0.0

def run_trading_cycle():
    se = st.session_state["stocks_engine"]
    now = dt.datetime.now(IST)
    is_market_open = (MARKET_START <= now.time() <= MARKET_END) and (now.weekday() < 5)
    
    if not is_market_open and se["running"]: return

    top_10, btst = fetch_stocks_data()
    if top_10: se["top_10"] = top_10
    if btst: se["btst_picks"] = btst

    # BUY
    if se["running"] and se["cash"] > (se["capital"] * 0.1):
        targets = se["top_10"][:3] + se["btst_picks"][:2]
        for stock in targets:
            sym = stock["Symbol"]
            if not any(p['symbol'] == sym for p in se['portfolio']):
                price = get_live_price(sym)
                alloc = se["capital"] * 0.15
                if price > 0 and se["cash"] > alloc:
                    qty = int(alloc / price)
                    se["portfolio"].append({
                        "symbol": sym, "entry": price, "qty": qty, 
                        "target_pct": stock["Exp. Profit"], "type": stock["Type"]
                    })
                    se["cash"] -= qty * price
                    se["logs"].insert(0, f"💣 Bought {sym} @ {price}")

    # SELL
    for i, p in enumerate(se["portfolio"]):
        curr = get_live_price(p["symbol"])
        if curr > 0:
            pnl_pct = ((curr - p["entry"]) / p["entry"]) * 100
            is_target = pnl_pct >= p["target_pct"]
            is_sl = pnl_pct <= -3.0
            
            if is_target or is_sl:
                val = p['qty'] * curr
                pnl = val - (p['qty'] * p['entry'])
                se["cash"] += val
                se["history"].append({"date": now, "symbol": p["symbol"], "pnl": pnl})
                
                msg = f"🚨 *Stock Closed*\n{p['symbol']} PnL: ₹{pnl:.2f}"
                send_telegram_alert(msg)
                
                se["portfolio"].pop(i)

def show_bomb_stocks_page():
    st.title("💣 Bomb Stocks (Auto-Pilot)")
    st.caption("AI-Selected High Velocity Stocks • Updates every 5 mins")
    st_autorefresh(interval=300_000, key="stock_refresh")
    run_trading_cycle()
    
    se = st.session_state["stocks_engine"]
    
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1: 
        if st.button("🚀 Start Engine"): se["running"] = True
    with c2: 
        if st.button("⏹ Stop Engine"): se["running"] = False
    
    with c3:
        st.markdown(f"**Status:** {'RUNNING' if se['running'] else 'STOPPED'}")
        st.markdown(f"**Cash:** ₹{se['cash']:,.0f} | **Inv:** ₹{se['capital']-se['cash']:,.0f}")

    st.markdown("---")
    
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("🎯 Top 10 High Velocity")
        if se["top_10"]: 
            df = pd.DataFrame(se["top_10"])
            st.dataframe(df[['Symbol', 'Exp. Profit', 'Velocity']], use_container_width=True)
    with c_right:
        st.subheader("🌙 BTST Picks")
        if se["btst_picks"]:
            df = pd.DataFrame(se["btst_picks"])
            st.dataframe(df[['Symbol', 'Exp. Profit']], use_container_width=True)

    st.markdown("---")
    st.subheader("💼 Active Portfolio")
    if se["portfolio"]:
        for i, p in enumerate(se["portfolio"]):
            curr = get_live_price(p["symbol"])
            pnl = (curr - p["entry"]) * p["qty"]
            
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.write(p["symbol"]); c2.write(f"Entry: {p['entry']}"); c3.write(f"Qty: {p['qty']}")
            c4.markdown(f":{'green' if pnl>0 else 'red'}[₹{pnl:.2f}]")
            
            if c5.button("Sell", key=f"s_{i}"):
                se["cash"] += curr * p["qty"]
                se["portfolio"].pop(i)
                st.rerun()
    else: st.write("No active trades.")

    with st.expander("Logs", expanded=True):
        for log in se["logs"][:5]: st.text(log)

# Entry Point
def run_stocks_app():
    init_stock_state()
    show_bomb_stocks_page()
