import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import datetime as dt
import pytz
import time
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
AIROBOTS_URL = "https://airobots.streamlit.app/"
IST = pytz.timezone("Asia/Kolkata")
MARKET_START = dt.time(9, 30)
MARKET_END = dt.time(15, 30)

# ---------------------------------------------------------
# STATE MANAGEMENT
# ---------------------------------------------------------
def init_stock_state():
    if "stocks_engine" not in st.session_state:
        st.session_state["stocks_engine"] = {
            "running": False,
            "capital": 100000.0,
            "cash": 100000.0,
            "portfolio": [],     # Active holdings
            "history": [],       # Closed trades
            "logs": [],          # Activity logs
            "top_10": [],        # High Velocity Picks
            "btst_picks": [],    # BTST Picks
            "last_scan": None
        }

# ---------------------------------------------------------
# AI ALGORITHM & DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_stocks_data():
    """
    Scrapes AI Robots, categorizes into Top 10 Velocity and BTST.
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(AIROBOTS_URL, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None, None

        tables = pd.read_html(response.text)
        if not tables:
            return None, None

        velocity_list = []
        btst_list = []
        
        # Time Factors: 1=Intraday, 2=BTST, 7=Weekly, 30=Monthly
        time_factors = [1, 2, 7, 30] 
        
        for i, df in enumerate(tables):
            # Basic cleaning
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Helper to find columns
            def get_col(keywords):
                return next((c for c in df.columns if any(k in c for k in keywords)), None)

            symbol_col = get_col(['stock', 'symbol', 'ticker'])
            profit_col = get_col(['profit', 'return', 'tgt'])
            
            if symbol_col and profit_col:
                # Determine type based on table index
                # 0 -> Intraday, 1 -> BTST, 2 -> Weekly, etc.
                category = ["Intraday", "BTST", "Weekly", "Monthly"][i] if i < 4 else "Long Term"
                days = time_factors[i] if i < len(time_factors) else 30
                
                for _, row in df.iterrows():
                    try:
                        sym = str(row[symbol_col]).upper().replace(".NS", "").strip()
                        raw_profit = str(row[profit_col]).replace("%", "").strip()
                        profit_pct = float(raw_profit) if raw_profit.replace('.', '', 1).isdigit() else 0.0
                        
                        if profit_pct > 0 and profit_pct < 100:
                            item = {
                                "Symbol": sym,
                                "Type": category,
                                "Exp. Profit": profit_pct,
                                "Velocity": profit_pct / days
                            }
                            
                            # Add to lists
                            velocity_list.append(item)
                            if category == "BTST":
                                btst_list.append(item)
                    except:
                        continue

        # Rank Top 10 by Velocity
        top_10 = pd.DataFrame(velocity_list).sort_values(by="Velocity", ascending=False).head(10).to_dict('records') if velocity_list else []
        btst = pd.DataFrame(btst_list).sort_values(by="Exp. Profit", ascending=False).head(10).to_dict('records') if btst_list else []
        
        return top_10, btst
            
    except Exception as e:
        return None, None

def get_live_price(ticker):
    try:
        sym = f"{ticker}.NS"
        data = yf.Ticker(sym).history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return 0.0

# ---------------------------------------------------------
# TRADING ENGINE
# ---------------------------------------------------------
def run_trading_cycle():
    se = st.session_state["stocks_engine"]
    now = dt.datetime.now(IST)
    
    # 1. Check Market Hours (09:30 - 15:30)
    # Allow running if user manually forced it, or during market hours
    is_market_open = (MARKET_START <= now.time() <= MARKET_END) and (now.weekday() < 5)
    
    if not is_market_open and se["running"]:
        # Log periodically if closed
        if now.minute % 30 == 0 and now.second < 5:
            se["logs"].insert(0, f"[{now.strftime('%H:%M')}] 💤 Market Closed. Scanning Paused.")
        return

    # 2. Fetch Data
    top_10, btst = fetch_stocks_data()
    if top_10: se["top_10"] = top_10
    if btst: se["btst_picks"] = btst

    # 3. Buy Logic
    if se["running"] and se["cash"] > (se["capital"] * 0.1):
        # Combine lists to check for opportunities
        targets = se["top_10"][:3] + se["btst_picks"][:2]
        
        for stock in targets:
            sym = stock["Symbol"]
            if not any(p['symbol'] == sym for p in se['portfolio']):
                price = get_live_price(sym)
                alloc = se["capital"] * 0.15 # 15% per trade
                
                if price > 0 and se["cash"] > alloc:
                    qty = int(alloc / price)
                    if qty > 0:
                        cost = qty * price
                        se["portfolio"].append({
                            "symbol": sym, "entry": price, "qty": qty,
                            "target_pct": stock["Exp. Profit"], "type": stock["Type"],
                            "date": now
                        })
                        se["cash"] -= cost
                        se["logs"].insert(0, f"[{now.strftime('%H:%M')}] 💣 BOUGHT {sym} ({stock['Type']}): {qty} qty @ ₹{price}")

    # 4. Sell Logic
    for i, pos in enumerate(se["portfolio"]):
        curr = get_live_price(pos["symbol"])
        if curr > 0:
            pnl_pct = ((curr - pos["entry"]) / pos["entry"]) * 100
            
            # Simple Exit Rules
            is_target = pnl_pct >= pos["target_pct"]
            is_sl = pnl_pct <= -3.0
            
            if is_target or is_sl:
                val = pos['qty'] * curr
                pnl_amt = val - (pos['qty'] * pos['entry'])
                se["cash"] += val
                
                se["history"].append({
                    "date": now, "symbol": pos["symbol"],
                    "pnl": pnl_amt, "roi": pnl_pct
                })
                
                status = "✅ PROFIT" if pnl_amt > 0 else "❌ LOSS"
                se["logs"].insert(0, f"[{now.strftime('%H:%M')}] {status}: Sold {pos['symbol']} @ ₹{curr} ({pnl_pct:.2f}%)")
                se["portfolio"].pop(i)
                
                # Telegram Alert Access
                if "tg_token" in st.session_state and st.session_state["tg_token"]:
                    try:
                        token = st.session_state["tg_token"]
                        chat = st.session_state["tg_chat_id"]
                        msg = (f"🚨 *Stock Trade Closed*\n"
                               f"Symbol: {pos['symbol']}\n"
                               f"Profit: ₹{pnl_amt:.2f} ({pnl_pct:.2f}%)")
                        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat, "text": msg, "parse_mode": "Markdown"})
                    except: pass

# ---------------------------------------------------------
# UI PAGES
# ---------------------------------------------------------
def show_bomb_stocks_page():
    st.title("💣 Bomb Stocks (Auto-Pilot)")
    st.caption("AI-Selected High Velocity Stocks • Updates every 5 mins • Auto-Trades during Market Hours")
    
    # 5 Minute Refresh
    st_autorefresh(interval=300_000, key="stock_refresh")
    run_trading_cycle()
    
    se = st.session_state["stocks_engine"]
    
    # Engine Control
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🚀 Start Engine", type="primary", use_container_width=True):
            se["running"] = True
    with c2:
        if st.button("⏹ Stop Engine", use_container_width=True):
            se["running"] = False
    
    status_color = "green" if se["running"] else "red"
    with c3:
        st.markdown(f"**Status:** :{status_color}[{'RUNNING' if se['running'] else 'STOPPED'}]")
        st.markdown(f"**Cash:** ₹{se['cash']:,.0f} | **Invested:** ₹{se['capital'] - se['cash']:,.0f}")

    st.markdown("---")

    # 1. Top 10 Velocity Table
    st.subheader("🎯 Top 10 'High Velocity' Picks")
    if se["top_10"]:
        df = pd.DataFrame(se["top_10"])
        df['Exp. Profit'] = df['Exp. Profit'].apply(lambda x: f"{x:.2f}%")
        df['Velocity'] = df['Velocity'].apply(lambda x: f"{x:.2f}")
        st.dataframe(df[['Symbol', 'Type', 'Exp. Profit', 'Velocity']], use_container_width=True)
    else:
        st.info("Waiting for AI scan...")

    # 2. BTST Table
    st.subheader("🌙 BTST Picks (Buy Today Sell Tomorrow)")
    if se["btst_picks"]:
        df_btst = pd.DataFrame(se["btst_picks"])
        df_btst['Exp. Profit'] = df_btst['Exp. Profit'].apply(lambda x: f"{x:.2f}%")
        st.dataframe(df_btst[['Symbol', 'Exp. Profit']], use_container_width=True)
    else:
        st.caption("No specific BTST signals found yet.")

    st.markdown("---")

    # 3. Active Portfolio
    st.subheader("💼 Active Portfolio")
    if se["portfolio"]:
        h1, h2, h3, h4, h5, h6 = st.columns([1.5, 1, 1, 1, 1, 1])
        h1.write("**Stock**"); h2.write("**Entry**"); h3.write("**CMP**"); 
        h4.write("**Qty**"); h5.write("**PnL**"); h6.write("**Action**")
        
        for i, p in enumerate(se["portfolio"]):
            curr = get_live_price(p["symbol"])
            invested = p["entry"] * p["qty"]
            current = curr * p["qty"]
            pnl = current - invested
            
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1])
            c1.write(f"{p['symbol']} ({p['type']})")
            c2.write(f"₹{p['entry']:.1f}")
            c3.write(f"₹{curr:.1f}")
            c4.write(f"{p['qty']}")
            c5.markdown(f":{'green' if pnl>0 else 'red'}[₹{pnl:.1f}]")
            
            if c6.button("Sell", key=f"sell_bomb_{i}"):
                se["cash"] += current
                se["history"].append({"date": dt.datetime.now(), "symbol": p['symbol'], "pnl": pnl, "roi": 0})
                se["logs"].insert(0, f"🔴 Manual Sell: {p['symbol']}")
                se["portfolio"].pop(i)
                st.rerun()
    else:
        st.write("No active trades.")

    with st.expander("📜 Activity Log", expanded=True):
        for log in se["logs"][:10]:
            st.text(log)

# Entry point called by app.py
def run_stocks_app():
    init_stock_state()
    show_bomb_stocks_page()
