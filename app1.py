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
            "top_10": [],        # Current best picks
            "last_scan": None
        }

# ---------------------------------------------------------
# AI ALGORITHM & DATA FETCHING
# ---------------------------------------------------------
@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_and_rank_stocks():
    """
    Scrapes AI Robots, normalizes data, and ranks stocks 
    based on 'Maximum Profit in Minimum Time' (Velocity Score).
    """
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(AIROBOTS_URL, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return None

        # Parse all tables found on the page
        tables = pd.read_html(response.text)
        if not tables:
            return None

        master_list = []
        
        # We assume tables might represent different timeframes (Intraday, BTST, etc.)
        # We assign a 'Time Factor' (days) to normalize profit velocity.
        # 1 = Intraday, 2 = BTST, 7 = Weekly, 30 = Monthly
        
        time_factors = [1, 2, 7, 30] 
        
        for i, df in enumerate(tables):
            # Clean columns
            df.columns = [str(c).lower().strip() for c in df.columns]
            
            # Identify key columns dynamically
            symbol_col = next((c for c in df.columns if 'stock' in c or 'symbol' in c), None)
            profit_col = next((c for c in df.columns if 'profit' in c or 'return' in c or 'tgt' in c), None)
            
            if symbol_col and profit_col:
                # Determine timeframe weight (fallback to Monthly if many tables)
                days = time_factors[i] if i < len(time_factors) else 30
                
                for _, row in df.iterrows():
                    try:
                        sym = str(row[symbol_col]).upper().replace(".NS", "").strip()
                        # Extract numeric profit (handle strings like "10%")
                        raw_profit = str(row[profit_col]).replace("%", "").strip()
                        profit_pct = float(raw_profit) if raw_profit.replace('.', '', 1).isdigit() else 0.0
                        
                        # ALGO: Velocity Score = Profit % / Days
                        # Penalize unrealistically high profits (>50% in intraday is likely an error/circuit)
                        if profit_pct > 0 and profit_pct < 50:
                            velocity = profit_pct / days
                            master_list.append({
                                "Symbol": sym,
                                "Type": ["Intraday", "BTST", "Weekly", "Monthly"][i] if i < 4 else "Long Term",
                                "Exp. Profit": profit_pct,
                                "Velocity Score": velocity
                            })
                    except:
                        continue

        # Sort by Velocity Score (High to Low)
        ranked_df = pd.DataFrame(master_list)
        if not ranked_df.empty:
            ranked_df = ranked_df.sort_values(by="Velocity Score", ascending=False).head(10)
            return ranked_df.to_dict('records')
            
    except Exception as e:
        return None
    return None

def get_live_price(ticker):
    """Get live price for Indian stocks (adds .NS)"""
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
    
    # 1. Check Market Hours
    if not (MARKET_START <= now.time() <= MARKET_END):
        if now.second < 5: # Log once per minute to avoid spam
            se["logs"].insert(0, f"[{now.strftime('%H:%M')}] 💤 Market Closed. Engine Sleeping.")
        return

    # 2. Fetch Top 10 "Bomb" Stocks
    top_picks = fetch_and_rank_stocks()
    if top_picks:
        se["top_10"] = top_picks
    else:
        # Fallback if scraping fails
        return 

    # 3. Buy Logic (Auto-Pilot)
    # Strategy: Buy top 3 available bombs if we have cash
    if se["running"] and se["cash"] > (se["capital"] * 0.1):
        for stock in se["top_10"][:3]: # Focus on Top 3 highest velocity
            sym = stock["Symbol"]
            
            # Check if already owned
            if not any(p['symbol'] == sym for p in se['portfolio']):
                price = get_live_price(sym)
                
                # Allocate 20% capital per trade
                allocation = se["capital"] * 0.2
                
                if price > 0 and se["cash"] > allocation:
                    qty = int(allocation / price)
                    if qty > 0:
                        cost = qty * price
                        
                        # Add to Portfolio
                        se["portfolio"].append({
                            "symbol": sym,
                            "entry": price,
                            "qty": qty,
                            "target_pct": stock["Exp. Profit"], # AI Target
                            "type": stock["Type"],
                            "time": now.strftime('%H:%M')
                        })
                        se["cash"] -= cost
                        se["logs"].insert(0, f"[{now.strftime('%H:%M')}] 💣 BOUGHT {sym} ({stock['Type']}): {qty} qty @ ₹{price}")

    # 4. Sell Logic (Target or Stop Loss)
    for i, pos in enumerate(se["portfolio"]):
        curr = get_live_price(pos["symbol"])
        if curr > 0:
            pnl_pct = ((curr - pos["entry"]) / pos["entry"]) * 100
            
            # AI Exit Strategy:
            # 1. Target Hit (based on AI Robots prediction)
            # 2. Stop Loss (hardcoded at -2% for safety)
            # 3. Intraday Close (if time > 3:15 PM)
            
            is_target = pnl_pct >= pos["target_pct"]
            is_sl = pnl_pct <= -2.0
            is_eod = (pos["type"] == "Intraday" and now.time() >= dt.time(15, 15))
            
            if is_target or is_sl or is_eod:
                val = pos['qty'] * curr
                pnl_amt = val - (pos['qty'] * pos['entry'])
                
                se["cash"] += val
                se["history"].append({
                    "date": now.strftime('%Y-%m-%d %H:%M'),
                    "symbol": pos["symbol"],
                    "pnl": pnl_amt,
                    "roi": pnl_pct,
                    "reason": "Target" if is_target else ("SL" if is_sl else "EOD")
                })
                
                status = "✅ PROFIT" if pnl_amt > 0 else "❌ LOSS"
                se["logs"].insert(0, f"[{now.strftime('%H:%M')}] {status}: Sold {pos['symbol']} @ ₹{curr} ({pnl_pct:.2f}%)")
                se["portfolio"].pop(i)
                
                # Send Telegram (if configured in main app)
                # We access main session state keys safely
                if "tg_token" in st.session_state and st.session_state["tg_token"]:
                    msg = (f"🚨 *Stock Trade Closed*\n"
                           f"Symbol: {pos['symbol']}\n"
                           f"Profit: ₹{pnl_amt:.2f} ({pnl_pct:.2f}%)\n"
                           f"Reason: {'Target Hit' if is_target else 'Stop Loss'}")
                    # We assume a send function exists or ignore if local
                    pass 

# ---------------------------------------------------------
# UI PAGES
# ---------------------------------------------------------
def show_bomb_stocks_page():
    st.title("💣 Bomb Stocks (Paper Trading)")
    st.caption("AI-Selected High Velocity Stocks • Auto-Refreshes every 5 mins")
    
    # Refresh logic (300s = 5 mins)
    st_autorefresh(interval=300_000, key="stock_refresh")
    
    # Run logic on every refresh
    run_trading_cycle()
    
    se = st.session_state["stocks_engine"]
    
    # 1. Engine Control
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        if st.button("🚀 Start Engine", type="primary", use_container_width=True):
            se["running"] = True
    with c2:
        if st.button("⏹ Stop Engine", use_container_width=True):
            se["running"] = False
    
    status_color = "green" if se["running"] else "red"
    status_text = "RUNNING" if se["running"] else "STOPPED"
    with c3:
        st.markdown(f"**Status:** :{status_color}[{status_text}]")
        st.markdown(f"**Cash:** ₹{se['cash']:,.2f} | **Invested:** ₹{se['capital'] - se['cash']:,.2f}")

    st.markdown("---")

    # 2. Top 10 Table
    st.subheader("🎯 Top 10 'Bomb' Picks (Live AI)")
    if se["top_10"]:
        df = pd.DataFrame(se["top_10"])
        # Formatting for display
        df['Exp. Profit'] = df['Exp. Profit'].apply(lambda x: f"{x:.2f}%")
        df['Velocity Score'] = df['Velocity Score'].apply(lambda x: f"{x:.2f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Waiting for market data / AI scan...")

    st.markdown("---")

    # 3. Active Portfolio
    st.subheader("💼 Active Portfolio")
    if se["portfolio"]:
        # Totals
        cur_val_sum = 0
        inv_sum = 0
        
        # Headers
        h1, h2, h3, h4, h5, h6 = st.columns([1.5, 1, 1, 1, 1, 1])
        h1.write("**Stock**"); h2.write("**Buy Price**"); h3.write("**CMP**"); 
        h4.write("**Qty**"); h5.write("**PnL**"); h6.write("**Action**")
        
        for i, p in enumerate(se["portfolio"]):
            curr = get_live_price(p["symbol"])
            invested = p["entry"] * p["qty"]
            current = curr * p["qty"]
            pnl = current - invested
            
            cur_val_sum += current
            inv_sum += invested
            
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1])
            c1.write(f"{p['symbol']} ({p['type']})")
            c2.write(f"₹{p['entry']:.1f}")
            c3.write(f"₹{curr:.1f}")
            c4.write(f"{p['qty']}")
            c5.markdown(f":{'green' if pnl>0 else 'red'}[₹{pnl:.1f}]")
            
            if c6.button("Sell", key=f"sell_bomb_{i}"):
                se["cash"] += current
                se["history"].append({"date": dt.datetime.now().strftime('%Y-%m-%d'), "symbol": p['symbol'], "pnl": pnl, "roi": 0})
                se["logs"].insert(0, f"🔴 Manual Sell: {p['symbol']}")
                se["portfolio"].pop(i)
                st.rerun()
        
        # Summary
        st.info(f"💰 **Total Invested:** ₹{inv_sum:,.0f}  |  **Current Value:** ₹{cur_val_sum:,.0f}  |  **Net PnL:** ₹{cur_val_sum - inv_sum:,.0f}")
        
    else:
        st.write("No active trades.")

    # 4. Logs
    with st.expander("📜 Activity Log", expanded=True):
        for log in se["logs"][:10]:
            st.text(log)

# ---------------------------------------------------------
# MAIN ENTRY POINT
# ---------------------------------------------------------
def run_stocks_app():
    init_stock_state()
    
    # We use the sub-page logic passed from app.py, 
    # but here we just handle the view rendering.
    # The navigation is controlled by the Radio in app.py
    # We just need to know WHICH page to render.
    
    # Since app.py calls this function, we can check st.session_state or just assume
    # we render the main "Bomb Stocks" page here.
    
    show_bomb_stocks_page()
