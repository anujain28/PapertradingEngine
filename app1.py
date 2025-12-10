import streamlit as st
import pandas as pd
import requests
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
import time
from streamlit_autorefresh import st_autorefresh

# ---------------------------------------------------------
# CONFIG & STATE INITIALIZATION
# ---------------------------------------------------------
AIROBOTS_URL = "https://airobots.streamlit.app/"

def init_stock_state():
    if "stock_autopilot" not in st.session_state:
        st.session_state["stock_autopilot"] = {
            "running": False,
            "capital": 100000.0, # Default 1 Lakh INR
            "cash": 100000.0,
            "portfolio": [],     # List of holdings
            "history": [],       # Closed trades
            "logs": [],
            "last_scan_time": None
        }

# ---------------------------------------------------------
# DATA FUNCTIONS
# ---------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_top_stocks_from_ai():
    """Scrapes the AI Robots app for the top table."""
    try:
        # User-Agent header often helps avoid bot detection
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(AIROBOTS_URL, headers=headers, timeout=10)
        
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            if tables:
                df = tables[0] # Assume first table is the main signal table
                # Clean column names
                df.columns = [c.lower() for c in df.columns]
                return df.head(10) # Return top 10 for analysis
    except Exception as e:
        return None
    return None

def get_live_stock_price(ticker):
    """Fetches live price from NSE (via Yahoo Finance). Adds .NS if missing."""
    clean_ticker = ticker.upper().replace(".NS", "").strip()
    ns_ticker = f"{clean_ticker}.NS"
    try:
        data = yf.Ticker(ns_ticker).history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return 0.0

@st.cache_data(ttl=600)
def get_stock_history(ticker, period="1mo"):
    clean_ticker = ticker.upper().replace(".NS", "").strip() + ".NS"
    try:
        return yf.Ticker(clean_ticker).history(period=period)
    except:
        return None

# ---------------------------------------------------------
# PAGE: STOCKS DASHBOARD
# ---------------------------------------------------------
def show_stocks_dashboard():
    st.title("📈 Indian Stocks Dashboard (NSE)")
    st_autorefresh(interval=60000, key="stock_dash_refresh")

    # 1. AI Analysis Table
    st.subheader("🤖 AI Robots: Top Picks")
    ai_data = fetch_top_stocks_from_ai()
    
    if ai_data is not None:
        st.dataframe(ai_data, use_container_width=True)
        # Extract stock names for the dropdown
        # Assuming the column name is 'symbol' or 'stock' or 'ticker'
        possible_cols = [c for c in ai_data.columns if 'symbol' in c or 'stock' in c or 'name' in c]
        stock_list = ai_data[possible_cols[0]].tolist() if possible_cols else ["RELIANCE", "TCS", "INFY", "HDFCBANK", "TATAMOTORS"]
    else:
        st.warning("Could not fetch data from AI Robots. Using default Nifty 50 list.")
        stock_list = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN", "TATAMOTORS", "ITC", "ADANIENT"]

    st.markdown("---")

    # 2. Detailed Charting
    c1, c2 = st.columns([3, 1])
    with c1:
        selected_stock = st.selectbox("Select Stock to Analyze", stock_list)
    with c2:
        period = st.selectbox("Timeframe", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "ytd", "max"], index=2)

    st.subheader(f"📊 {selected_stock} Price Chart")
    
    hist = get_stock_history(selected_stock, period)
    if hist is not None and not hist.empty:
        curr_price = hist['Close'].iloc[-1]
        
        # Color chart based on trend
        start_price = hist['Close'].iloc[0]
        color = 'green' if curr_price >= start_price else 'red'
        
        st.metric("Current Price", f"₹{curr_price:,.2f}", 
                 delta=f"{(curr_price - start_price)/start_price * 100:.2f}%")

        fig = go.Figure(data=[go.Candlestick(x=hist.index,
                        open=hist['Open'], high=hist['High'],
                        low=hist['Low'], close=hist['Close'])])
        
        fig.update_layout(
            height=500, margin=dict(l=0,r=0,t=0,b=0),
            plot_bgcolor='white', paper_bgcolor='white',
            font=dict(color='black'),
            xaxis=dict(showgrid=True, gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=True, gridcolor='#f0f0f0')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error(f"Data not available for {selected_stock}.NS")

# ---------------------------------------------------------
# PAGE: STOCK AUTO-PILOT
# ---------------------------------------------------------
def show_stock_autopilot():
    st.title("🚀 Stocks Auto-Pilot")
    st.caption("Automatically buys the 'Top 5' stocks from AI Robots analysis and manages the portfolio.")
    st_autorefresh(interval=30000, key="stock_ap_refresh")
    
    sap = st.session_state["stock_autopilot"]

    # --- CONFIGURATION ---
    if not sap["running"]:
        st.subheader("⚙️ Engine Configuration")
        c1, c2 = st.columns(2)
        capital = c1.number_input("Total Capital (INR)", min_value=10000.0, value=100000.0, step=5000.0)
        
        if c2.button("🚀 Start Stock Engine", type="primary"):
            sap["running"] = True
            sap["capital"] = capital
            sap["cash"] = capital
            sap["logs"].append(f"[{dt.datetime.now().strftime('%H:%M')}] Engine Started. Capital: ₹{capital:,.2f}")
            st.rerun()
    else:
        # --- RUNNING STATS ---
        st.success("✅ Stock Engine is Active: Scanning AI Robots...")
        
        # Calculate Metrics
        invested_val = sum([p['qty'] * get_live_stock_price(p['symbol']) for p in sap['portfolio']])
        total_val = sap['cash'] + invested_val
        pnl = total_val - sap['capital']
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Portfolio Value", f"₹{total_val:,.2f}")
        m2.metric("Cash Balance", f"₹{sap['cash']:,.2f}")
        m3.metric("Invested Amount", f"₹{invested_val:,.2f}")
        m4.metric("Total PnL", f"₹{pnl:,.2f}", delta_color="normal")
        
        st.markdown("---")
        
        # --- ENGINE LOGIC ---
        # 1. Fetch Top 5 if we have cash
        if sap['cash'] > (sap['capital'] * 0.1): # Keep 10% buffer
            ai_df = fetch_top_stocks_from_ai()
            
            if ai_df is not None and not ai_df.empty:
                # Find column with symbol name
                cols = [c for c in ai_df.columns if 'symbol' in c or 'stock' in c]
                if cols:
                    top_picks = ai_df[cols[0]].head(5).tolist()
                    
                    # Diversify: Allocate 20% of INITIAL capital per stock max
                    max_alloc = sap['capital'] * 0.2
                    
                    for stock in top_picks:
                        # Check if already owned
                        owned = any(p['symbol'] == stock for p in sap['portfolio'])
                        if not owned and sap['cash'] > max_alloc:
                            price = get_live_stock_price(stock)
                            if price > 0:
                                qty = int(max_alloc / price)
                                if qty > 0:
                                    cost = qty * price
                                    
                                    # EXECUTE BUY
                                    sap['portfolio'].append({
                                        "symbol": stock,
                                        "entry": price,
                                        "qty": qty,
                                        "date": dt.datetime.now()
                                    })
                                    sap['cash'] -= cost
                                    sap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M')}] 🟢 BOUGHT {stock}: {qty} qty @ ₹{price}")
        
        # 2. Check for Exits (Simple Trail or 2% Profit Simulation)
        for i, pos in enumerate(sap['portfolio']):
            curr = get_live_stock_price(pos['symbol'])
            if curr > 0:
                roi = (curr - pos['entry']) / pos['entry'] * 100
                # Take profit at 2% or Stop Loss at 5% (Simulation Logic)
                if roi >= 2.0 or roi <= -5.0:
                    val = pos['qty'] * curr
                    pnl_trade = val - (pos['qty'] * pos['entry'])
                    
                    # Close Trade
                    sap['cash'] += val
                    sap['history'].append({
                        "date": dt.datetime.now(),
                        "symbol": pos['symbol'],
                        "pnl": pnl_trade,
                        "roi": roi
                    })
                    sap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M')}] 🔴 SOLD {pos['symbol']}: ROI {roi:.2f}% (PnL ₹{pnl_trade:.0f})")
                    sap['portfolio'].pop(i)
                    st.rerun()

        # --- UI: ACTIVE POSITIONS ---
        st.subheader("💼 Active Holdings")
        if sap['portfolio']:
            # Header
            c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1])
            c1.markdown("**Stock**"); c2.markdown("**Entry**"); c3.markdown("**CMP**")
            c4.markdown("**Qty**"); c5.markdown("**PnL**"); c6.markdown("**Action**")
            
            for i, p in enumerate(sap['portfolio']):
                curr = get_live_stock_price(p['symbol'])
                trade_val = curr * p['qty']
                cost_val = p['entry'] * p['qty']
                unrealized = trade_val - cost_val
                
                c1, c2, c3, c4, c5, c6 = st.columns([1.5, 1, 1, 1, 1, 1])
                c1.write(p['symbol'])
                c2.write(f"₹{p['entry']:.1f}")
                c3.write(f"₹{curr:.1f}")
                c4.write(f"{p['qty']}")
                
                clr = "green" if unrealized >= 0 else "red"
                c5.markdown(f":{clr}[₹{unrealized:.1f}]")
                
                if c6.button("Sell", key=f"sell_stk_{i}"):
                    sap['cash'] += trade_val
                    sap['history'].append({
                        "date": dt.datetime.now(),
                        "symbol": p['symbol'],
                        "pnl": unrealized,
                        "roi": (unrealized/cost_val)*100
                    })
                    sap['logs'].insert(0, f"[{dt.datetime.now().strftime('%H:%M')}] 🔴 MANUAL SELL {p['symbol']}")
                    sap['portfolio'].pop(i)
                    st.rerun()
        else:
            st.info("Scanning for high-potential stocks...")

        # Logs
        with st.expander("📝 Activity Log", expanded=True):
            for log in sap['logs'][:5]:
                st.text(log)

        if st.button("⏹ Stop Engine"):
            sap["running"] = False
            st.rerun()

# ---------------------------------------------------------
# MAIN ENTRY POINT FOR APP.PY
# ---------------------------------------------------------
def run_stocks_app():
    init_stock_state()
    
    # Sub-Navigation for Stocks
    page = st.sidebar.radio("Stocks Menu", ["Dashboard", "Auto-Pilot"])
    
    if page == "Dashboard":
        show_stocks_dashboard()
    elif page == "Auto-Pilot":
        show_stock_autopilot()
