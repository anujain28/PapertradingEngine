            # app.py
import os
import time
import threading
import datetime as dt
import sqlite3
from typing import Dict, List, Optional

import configparser
import pytz
import requests
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh  # pip install streamlit-autorefresh
from telegram.ext import Application
from nsepython import nse_quote_ltp  # NSEPython for LTP [web:73][web:75]
from crypto_bot import (
    init_crypto_state, get_binance_client, crypto_trading_loop,
    save_binance_config, load_binance_config, get_crypto_positions,
    get_crypto_trades
)

# ---------------------------
# PAGE CONFIG + GLOBAL STYLE
# ---------------------------
st.set_page_config(page_title="AI Paper Trading", layout="wide")


def apply_custom_style():
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            font-family: Inter, system-ui, -apple-system, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif !important;
        }
        .stApp {
            background-color: #f3f4f6 !important;  /* soft grey */
            color: #0f172a !important;
        }
        /* Center main content and control width */
        div.block-container {
            max-width: 1100px;
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            margin: 0 auto;
        }
        /* Metrics / cards spacing */
.stMetric {
    background-color: #ffffff !important;
    border: 2px solid #e0e7ff !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08) !important;
}
    .stMetric label {
    font-size: 0.95rem !important;
    color: #64748b !important;
    font-weight: 600 !important;
    margin-bottom: 6px !important;
}
    .stMetric [data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    color: #1e293b !important;
    font-weight: 700 !important;
}        /* Tables */
        div[data-testid="stDataFrame"] table {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border-collapse: collapse !important;
        }
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {
            border-color: #e5e7eb !important;
            font-size: 0.9rem !important;
            padding: 6px 10px !important;
        }
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #f9fafb !important;
        }
        /* Buttons */
        .stButton>button {
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------
# CONFIG
# ---------------------------
IST = pytz.timezone("Asia/Kolkata")
TRADING_START = dt.time(9, 30)
TRADING_END = dt.time(15, 30)

START_CAPITAL = 100000.0
MAX_UTILIZATION = 0.60  # 60% of capital in the market

TRAIL_PCT_DEFAULT = 0.02  # 2% trailing stop

AIROBOTS_URL = "https://airobots.streamlit.app/"
DB_PATH = "paper_trades.db"
CONFIG_PATH = "telegram_config.ini"


# ---------------------------
# TELEGRAM CONFIG (LOCAL FILE)
# ---------------------------
def load_telegram_config():
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_PATH):
        cfg.read(CONFIG_PATH)
    token = ""
    chat_id = ""
    if "telegram" in cfg:
        token = cfg["telegram"].get("token", "")
        chat_id = cfg["telegram"].get("chat_id", "")
    return token, chat_id


def save_telegram_config(token: str, chat_id: str):
    cfg = configparser.ConfigParser()
    if os.path.exists(CONFIG_PATH):
        cfg.read(CONFIG_PATH)
    if "telegram" not in cfg:
        cfg["telegram"] = {}
    cfg["telegram"]["token"] = token.strip()
    cfg["telegram"]["chat_id"] = chat_id.strip()
    with open(CONFIG_PATH, "w") as f:
        cfg.write(f)


TELEGRAM_TOKEN, TELEGRAM_CHAT_ID = load_telegram_config()


# ---------------------------
# GLOBAL STATE (IN MEMORY)
# ---------------------------
if "state" not in st.session_state:
    st.session_state["state"] = {
        "capital": START_CAPITAL,
        "equity": START_CAPITAL,
        "positions": {},   # symbol -> position dict
    }

if "engine_status" not in st.session_state:
    st.session_state["engine_status"] = "Idle"

if "engine_running" not in st.session_state:
    st.session_state["engine_running"] = False

if "loop_started" not in st.session_state:
    st.session_state["loop_started"] = False

if "telegram_started" not in st.session_state:
    st.session_state["telegram_started"] = False

if "report_time" not in st.session_state:
    st.session_state["report_time"] = dt.time(16, 0)  # default 16:00 IST

# store last picked top 5
if "last_top5" not in st.session_state:
    st.session_state["last_top5"] = []

# Initialize crypto state
init_crypto_state()


# ---------------------------
# DB INIT
# ---------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            side TEXT,
            qty INTEGER,
            price REAL,
            timestamp TEXT,
            pnl REAL
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            timestamp TEXT,
            equity REAL,
            capital REAL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


# ---------------------------
# MARKET DATA / AIROBOTS
# ---------------------------
def fetch_top5_from_airobots() -> List[Dict]:
    """
    Read first HTML table from AI Robots app,
    pick top 5 by 'Profit %'. Adjust column names as per actual app.
    """
    try:
        resp = requests.get(AIROBOTS_URL, timeout=15)
        resp.raise_for_status()
        tables = pd.read_html(resp.text)
        if not tables:
            return []
        df = tables[0].copy()
        df.columns = [str(c).strip() for c in df.columns]

        profit_col = None
        symbol_col = None
        for c in df.columns:
            lc = c.lower()
            if "profit" in lc and "%" in lc:
                profit_col = c
            if "symbol" in lc or "stock" in lc or "ticker" in lc:
                symbol_col = c

        if profit_col is None or symbol_col is None:
            return []

        df = df.sort_values(profit_col, ascending=False).head(5)
        df.rename(columns={symbol_col: "Symbol", profit_col: "Profit %"}, inplace=True)
        return df.to_dict(orient="records")
    except Exception:
        return []


def normalize_symbol_for_nse(symbol: str) -> str:
    """
    Convert incoming symbol to NSEPython-friendly ticker.
    Adjust this as per your AI Robots symbol format.
    Examples:
      'NSE:TCS' -> 'TCS'
      'TCS-EQ'  -> 'TCS'
    """
    s = symbol.strip().upper()
    if ":" in s:
        s = s.split(":", 1)[1]
    if s.endswith("-EQ"):
        s = s.replace("-EQ", "")
    return s


def get_ltp(symbol: str) -> Optional[float]:
    """
    Get last traded price from NSE using NSEPython.
    Assumes symbol is an NSE EQ symbol (e.g. TCS, HDFCBANK). [web:71][web:73]
    """
    try:
        nse_sym = normalize_symbol_for_nse(symbol)
        ltp = nse_quote_ltp(nse_sym)
        if ltp is None:
            return None
        return float(ltp)
    except Exception:
        return None


# ---------------------------
# TRADING ENGINE
# ---------------------------
def recompute_equity():
    state = st.session_state["state"]
    mtm = 0.0
    for pos in state["positions"].values():
        mtm += pos["qty"] * pos["last_price"]
    state["equity"] = state["capital"] + mtm


def persist_trades(trades: List[Dict]):
    if not trades:
        return
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(trades)
    df.to_sql("trades", conn, if_exists="append", index=False)
    conn.close()


def persist_equity_snapshot(now: dt.datetime):
    state = st.session_state["state"]
    conn = sqlite3.connect(DB_PATH)
    row = {
        "date": now.date().isoformat(),
        "timestamp": now.isoformat(),
        "equity": state["equity"],
        "capital": state["capital"],
    }
    pd.DataFrame([row]).to_sql("equity_snapshots", conn, if_exists="append", index=False)
    conn.close()


def realize_profit(symbol: str, exit_price: float, now: dt.datetime, batch_trades: List[Dict]):
    state = st.session_state["state"]
    if symbol not in state["positions"]:
        return
    pos = state["positions"][symbol]
    qty = pos["qty"]
    entry = pos["entry_price"]
    pnl = (exit_price - entry) * qty

    state["capital"] += qty * exit_price
    batch_trades.append(
        {
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "price": exit_price,
            "timestamp": now.isoformat(),
            "pnl": pnl,
        }
    )
    del state["positions"][symbol]


def update_positions_and_trails(now: dt.datetime, batch_trades: List[Dict]):
    state = st.session_state["state"]
    to_exit = []

    for sym, pos in list(state["positions"].items()):
        ltp = get_ltp(sym)
        if ltp is None:
            # If no price, carry forward without change
            continue

        pos["last_price"] = ltp
        if ltp > pos["max_price"]:
            pos["max_price"] = ltp

        trail_pct = pos.get("trail_pct", TRAIL_PCT_DEFAULT)
        trail_stop = pos["max_price"] * (1 - trail_pct)
        # No loss rule: only exit if ltp >= entry_price
        min_exit_price = max(trail_stop, pos["entry_price"])
        if ltp <= trail_stop and ltp >= pos["entry_price"]:
                to_exit.append((sym, ltp))
                for sym, exit_price in to_exit:
                        realize_profit(sym, exit_price, now, batch_trades)
def rebalance_entries(top5: List[Dict], now: dt.datetime, batch_trades: List[Dict]):
    if not top5:
        return
    state = st.session_state["state"]
    usable_capital = START_CAPITAL * MAX_UTILIZATION
    invested_value = sum(p["qty"] * p["last_price"] for p in state["positions"].values())
    available_for_new = max(0.0, usable_capital - invested_value)

    if available_for_new <= 0:
        return

    per_stock_alloc = available_for_new / len(top5)

    for row in top5:
        sym = str(row.get("Symbol")).strip()
        if not sym:
            continue
        if sym in state["positions"]:
            continue

        ltp = get_ltp(sym)
        if ltp is None or ltp <= 0:
            continue

        qty = int(per_stock_alloc // ltp)
        if qty <= 0:
            continue

        cost = qty * ltp
        if state["capital"] < cost:
            continue

        state["capital"] -= cost
        pos = {
            "symbol": sym,
            "qty": qty,
            "entry_price": ltp,
            "last_price": ltp,
            "max_price": ltp,
            "trail_pct": TRAIL_PCT_DEFAULT,
            "open_date": now.date().isoformat(),
        }
        state["positions"][sym] = pos
        batch_trades.append(
            {
                "symbol": sym,
                "side": "BUY",
                "qty": qty,
                "price": ltp,
                "timestamp": now.isoformat(),
                "pnl": 0.0,
            }
        )


def run_trading_cycle(now: dt.datetime):
    """
    One full cycle:
    - Fetch top 5 from AI Robots
    - Update prices and trailing stops
    - Enter new positions if capital available
    - Persist trades + equity snapshot
    """
    batch_trades: List[Dict] = []

    st.session_state["engine_status"] = (
        f"Running cycle at {now.strftime('%H:%M:%S')} – fetching top 5 from AI Robots..."
    )
    top5 = fetch_top5_from_airobots()
    st.session_state["last_top5"] = top5  # store for sidebar display

    st.session_state["engine_status"] = (
        f"Updating positions & trailing stops at {now.strftime('%H:%M:%S')}..."
    )
    update_positions_and_trails(now, batch_trades)

    st.session_state["engine_status"] = (
        f"Evaluating new entries for {len(top5)} candidates..."
    )
    rebalance_entries(top5, now, batch_trades)

    recompute_equity()
    persist_trades(batch_trades)
    persist_equity_snapshot(now)
    st.session_state["engine_status"] = "Cycle complete. Waiting for next run."


def trading_loop():
    """
    Background loop; respects engine_running flag.
    Runs only on weekdays between 9:30 and 15:30 IST.
    Uses .get() to avoid AttributeError if keys not initialized. [web:61][web:69]
    """
    while True:
        engine_running = st.session_state.get("engine_running", False)
        if not engine_running:
            st.session_state["engine_status"] = "Engine stopped."
            time.sleep(2)
            continue

        now = dt.datetime.now(IST)
        if now.weekday() < 5 and TRADING_START <= now.time() <= TRADING_END:
            run_trading_cycle(now)
        else:
            st.session_state["engine_status"] = (
                "Engine running but outside market hours (9:30–15:30 IST)."
            )
        time.sleep(120)


# ---------------------------
# PNL LOADING
# ---------------------------
def load_pnl_frames():
    conn = sqlite3.connect(DB_PATH)
    trades = pd.read_sql("SELECT * FROM trades", conn)
    conn.close()

    if trades.empty:
        daily = pd.DataFrame(columns=["date", "pnl"])
        weekly = pd.DataFrame(columns=["week", "pnl"])
        monthly = pd.DataFrame(columns=["month", "pnl"])
        return daily, weekly, monthly

    trades["timestamp"] = pd.to_datetime(trades["timestamp"])
    trades["date"] = trades["timestamp"].dt.date

    daily = trades.groupby("date")["pnl"].sum().reset_index()

    weekly = (
        trades.groupby(trades["timestamp"].dt.to_period("W"))["pnl"]
        .sum()
        .reset_index()
    )
    weekly.rename(columns={"timestamp": "week"}, inplace=True)

    monthly = (
        trades.groupby(trades["timestamp"].dt.to_period("M"))["pnl"]
        .sum()
        .reset_index()
    )
    monthly.rename(columns={"timestamp": "month"}, inplace=True)

    return daily, weekly, monthly


def get_today_pnl(daily_df: pd.DataFrame) -> float:
    if daily_df.empty:
        return 0.0
    today = dt.datetime.now(IST).date()
    row = daily_df[daily_df["date"] == today]
    if row.empty:
        return 0.0
    return float(row["pnl"].sum())


# ---------------------------
# TELEGRAM SCHEDULER
# ---------------------------
async def send_telegram_report(context):
    daily, weekly, monthly = load_pnl_frames()
    today_pnl = get_today_pnl(daily)
    equity = st.session_state["state"]["equity"]
    msg = (
        f"Paper Trading Report\n"
        f"Date: {dt.datetime.now(IST).date().isoformat()}\n"
        f"Today's PnL: {today_pnl:.2f}\n"
        f"Equity: {equity:.2f}"
    )
    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)


async def _start_telegram_jobs():
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    rt = st.session_state["report_time"]
    app.job_queue.run_daily(
        send_telegram_report,
        time=dt.time(rt.hour, rt.minute, tzinfo=IST),
        days=(0, 1, 2, 3, 4),  # Mon–Fri
    )
    await app.initialize()
    await app.start()
    await app.updater.start_polling()


def start_telegram_scheduler_if_needed():
    if st.session_state.get("telegram_started", False):
        return
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return

    def runner():
        import asyncio
        asyncio.run(_start_telegram_jobs())

    threading.Thread(target=runner, daemon=True).start()
    st.session_state["telegram_started"] = True


# ---------------------------
# STREAMLIT UI
# ---------------------------
def show_paper_trading_page():
    st.title("📈 AI Paper Trading Engine")

    # Auto-refresh every 2 minutes
    st_autorefresh(interval=120_000, key="auto_refresh")

    # Engine status banner
    status = st.session_state.get("engine_status", "Idle")
    if status.lower().startswith("running"):
        st.info(f"🟢 Engine status: {status}")
    elif "stopped" in status.lower():
        st.warning(f"🛑 Engine status: {status}")
    else:
        st.info(f"ℹ️ Engine status: {status}")

    state = st.session_state["state"]

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Free Capital", f"₹{state['capital']:.2f}")
    with col2:
        st.metric("Equity (Cash + MTM)", f"₹{state['equity']:.2f}")

    positions = list(state["positions"].values())
    if positions:
        st.subheader("Open Positions")
        df_pos = pd.DataFrame(positions)
        st.dataframe(df_pos, use_container_width=True)
    else:
        st.info("No open positions currently. Engine will auto-scan and enter trades during market hours.")

    st.caption(
        "Engine runs automatically from 9:30 AM to 3:30 PM IST when started, "
        "fetches top 5 stocks from AI Robots, uses up to 60% of capital, "
        "trails profits, and never exits at a loss (positions are carried forward)."
    )


def show_pnl_page():
    st.title("📊 PNL Log")

    daily, weekly, monthly = load_pnl_frames()

    st.subheader("Daily PnL")
    st.dataframe(daily, use_container_width=True)

    st.subheader("Weekly PnL")
    st.dataframe(weekly, use_container_width=True)

    st.subheader("Monthly PnL")
    st.dataframe(monthly, use_container_width=True)


def show_crypto_page():
    st.title("🤖 Crypto Grid Trading Bot (24/7)")
    st.info("🟢 Binance Spot Grid Trading - ETH, BTC, SOL, ADA, XRP")
    
    # Binance API config
    with st.expander("⚙️ Binance API Configuration", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            api_key = st.text_input("API Key", type="password", help="Binance API Key")
        with col2:
            secret_key = st.text_input("Secret Key", type="password", help="Binance Secret Key")
        
        if st.button("Save Binance Credentials"):
            if api_key and secret_key:
                save_binance_config(api_key, secret_key)
                st.success("Binance credentials saved!")
            else:
                st.error("Please enter both API Key and Secret Key")
    
    # Bot status
    status = st.session_state.get("crypto_status", "Idle")
    st.metric("Bot Status", status)
    
    # Bot control
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start 24/7 Trading"):
            st.session_state["crypto_running"] = True
            st.session_state["crypto_status"] = "Starting grid trading on 5 coins..."
            st.success("Crypto bot started! Running 24/7...")
    with col2:
        if st.button("⏹ Stop Trading"):
            st.session_state["crypto_running"] = False
            st.session_state["crypto_status"] = "Bot stopped"
            st.warning("Crypto bot stopped.")
    
    # Positions
    st.subheader("📍 Open Grid Positions")
    positions = get_crypto_positions()
    if not positions.empty:
        st.dataframe(positions, use_container_width=True)
    else:
        st.info("No active grid positions yet. Start the bot to begin trading.")
    
    # Recent trades
    st.subheader("📊 Recent Grid Orders")
    trades = get_crypto_trades()
    if not trades.empty:
        st.dataframe(trades, use_container_width=True)
    else:
        st.info("No trades executed yet.")
    
    st.caption(
        "Grid Trading uses Binance Spot Grid features to automatically place buy/sell orders. "
        "The bot manages 5 cryptocurrencies (BTC, ETH, SOL, ADA, XRP) simultaneously, "
        "running 24/7 to capitalize on market volatility with minimal risk."
    )

def sidebar_config():
    st.sidebar.header("⚙️ Configuration")

    # Telegram config (local file)
    st.sidebar.subheader("Telegram settings (local)")

    token_input = st.sidebar.text_input(
        "Bot token",
        value=TELEGRAM_TOKEN,
        type="password",
        help="Saved in telegram_config.ini on this machine.",
    )
    chat_input = st.sidebar.text_input(
        "Chat ID",
        value=TELEGRAM_CHAT_ID,
        help="Your personal / group chat id for reports.",
    )

    if st.sidebar.button("💾 Save Telegram config"):
        save_telegram_config(token_input, chat_input)
        st.sidebar.success("Telegram config saved locally. Restart app to reload.")

    # Report time
    report_time_str = st.sidebar.text_input(
        "Telegram report time (HH:MM, IST)",
        value=f"{st.session_state['report_time'].hour:02d}:{st.session_state['report_time'].minute:02d}",
    )
    try:
        hh, mm = report_time_str.split(":")
        hh, mm = int(hh), int(mm)
        st.session_state["report_time"] = dt.time(hh, mm)
    except Exception:
        st.sidebar.warning("Invalid time format. Use HH:MM (e.g. 16:00).")

    if token_input and chat_input:
        st.sidebar.success("Telegram configured. Daily report will be sent on weekdays.")
    else:
        st.sidebar.info("Enter bot token & chat id above to enable Telegram reports.")

    # Engine control
    st.sidebar.subheader("Engine control")
    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.sidebar.button("▶️ Start engine"):
            st.session_state["engine_running"] = True
            st.session_state["engine_status"] = "Engine started. Waiting for market window."
    with col_b:
        if st.sidebar.button("⏹ Stop engine"):
            st.session_state["engine_running"] = False
            st.session_state["engine_status"] = "Engine stopped by user."

    # Show last picked top 5
    st.sidebar.subheader("Last top 5 from AI Robots")
    if st.session_state["last_top5"]:
        top5_df = pd.DataFrame(st.session_state["last_top5"])
        st.sidebar.dataframe(top5_df, use_container_width=True, height=220)
    else:
        st.sidebar.info("No scan results yet. Start engine and wait for first cycle.")


def main():
    apply_custom_style()
    sidebar_config()

    # Start engine loop once
    if not st.session_state.get("loop_started", False):
        t = threading.Thread(target=trading_loop, daemon=True)
        t.start()
        st.session_state["loop_started"] = True

        # Start crypto trading loop once
    if not st.session_state.get("crypto_loop_started", False):
        t_crypto = threading.Thread(target=crypto_trading_loop, daemon=True)
        t_crypto.start()
        st.session_state["crypto_loop_started"] = True

    # Start Telegram scheduler once
    start_telegram_scheduler_if_needed()

 page = st.sidebar.radio("Pages", ["Paper Trading", "PNL Log", "Crypto Bot"])

    if page == "Paper Trading":
        show_paper_trading_page()
    elif page == "PNL Log":
    show_pnl_page()
    elif page == "Crypto Bot":
        show_crypto_page()


if __name__ == "__main__":
    main()
