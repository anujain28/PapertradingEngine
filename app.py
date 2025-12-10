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
from nsepython import nse_quote_ltp  # NSEPython for LTP
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
    # All fonts black globally, tables white background
    st.markdown(
        """
        <style>
        html, body, [class*="css"]  {
            color: #000000 !important;
        }
        .stApp {
            color: #000000 !important;
        }
        div[data-testid="stDataFrame"] table {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        div[data-testid="stDataFrame"] th,
        div[data-testid="stDataFrame"] td {
            background-color: #ffffff !important;
            color: #000000 !important;
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
    Assumes symbol is an NSE EQ symbol (e.g. TCS, HDFCBANK).
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
                "pnl"
