# crypto_bot.py
import os
import time
import threading
import datetime as dt
import sqlite3
from typing import Dict, List, Optional, Tuple
import configparser
import pytz
import pandas as pd
import streamlit as st
from binance.client import Client
from binance.exceptions import BinanceAPIException
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# CONFIG
# ---------------------------
BINANCE_API_KEY_FILE = "binance_api_key.ini"
CRYPTO_DB_PATH = "crypto_trades.db"
GRID_COINS = ["BTC", "ETH", "SOL", "ADA", "XRP"]  # 5 coins
GRID_TRADING_PARAMS = {
    "BTC": {"investment": 1000, "grids": 10, "profit_percent": 0.5},
    "ETH": {"investment": 800, "grids": 10, "profit_percent": 0.5},
    "SOL": {"investment": 500, "grids": 10, "profit_percent": 0.5},
    "ADA": {"investment": 400, "grids": 10, "profit_percent": 0.5},
    "XRP": {"investment": 400, "grids": 10, "profit_percent": 0.5},
}

# ---------------------------
# BINANCE CONFIG
# ---------------------------
def load_binance_config():
    cfg = configparser.ConfigParser()
    if os.path.exists(BINANCE_API_KEY_FILE):
        cfg.read(BINANCE_API_KEY_FILE)
        api_key = cfg.get("binance", "api_key", fallback="")
        secret_key = cfg.get("binance", "secret_key", fallback="")
        return api_key, secret_key
    return "", ""

def save_binance_config(api_key: str, secret_key: str):
    cfg = configparser.ConfigParser()
    if not cfg.has_section("binance"):
        cfg.add_section("binance")
    cfg.set("binance", "api_key", api_key.strip())
    cfg.set("binance", "secret_key", secret_key.strip())
    with open(BINANCE_API_KEY_FILE, "w") as f:
        cfg.write(f)

def get_binance_client():
    api_key, secret_key = load_binance_config()
    if not api_key or not secret_key:
        return None
    try:
        client = Client(api_key, secret_key)
        client.ping()  # Test connection
        return client
    except Exception as e:
        logger.error(f"Binance connection error: {e}")
        return None

# ---------------------------
# DATABASE
# ---------------------------
def init_crypto_db():
    conn = sqlite3.connect(CRYPTO_DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crypto_grid_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT,
            side TEXT,
            quantity REAL,
            price REAL,
            timestamp TEXT,
            status TEXT,
            binance_order_id TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crypto_positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            coin TEXT UNIQUE,
            quantity REAL,
            entry_price REAL,
            current_price REAL,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crypto_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            coin TEXT,
            quantity REAL,
            value REAL,
            profit_loss REAL
        )
    """)
    conn.commit()
    conn.close()

init_crypto_db()

# ---------------------------
# GRID TRADING LOGIC
# ---------------------------
class GridTrader:
    def __init__(self, client: Client, coin: str, params: Dict):
        self.client = client
        self.coin = coin
        self.symbol = f"{coin}USDT"
        self.investment = params["investment"]
        self.grids = params["grids"]
        self.profit_percent = params["profit_percent"]
        self.grid_orders = []
        self.position = None
        self.last_price = 0

    def get_current_price(self) -> Optional[float]:
        try:
            ticker = self.client.get_symbol_ticker(symbol=self.symbol)
            return float(ticker["price"])
        except Exception as e:
            logger.error(f"Error getting price for {self.symbol}: {e}")
            return None

    def create_grid_orders(self) -> List[Dict]:
        """Create grid buy orders below current price."""
        current_price = self.get_current_price()
        if not current_price:
            return []
        
        self.last_price = current_price
        grid_spacing = current_price * 0.02  # 2% spacing
        quantity = self.investment / (current_price * self.grids)
        
        orders = []
        for i in range(self.grids):
            price = current_price - (grid_spacing * (i + 1))
            if price > 0:
                orders.append({
                    "coin": self.coin,
                    "side": "BUY",
                    "quantity": quantity,
                    "price": price,
                    "type": "LIMIT"
                })
        return orders

    def place_grid_orders(self) -> bool:
        """Place grid orders on Binance."""
        if not self.client:
            return False
        
        orders = self.create_grid_orders()
        placed = 0
        
        for order_params in orders:
            try:
                order = self.client.order_limit_buy(
                    symbol=self.symbol,
                    quantity=order_params["quantity"],
                    price=order_params["price"]
                )
                placed += 1
                logger.info(f"Placed order for {self.coin}: {order['orderId']}")
                
                # Save to DB
                conn = sqlite3.connect(CRYPTO_DB_PATH)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO crypto_grid_orders 
                    (coin, side, quantity, price, timestamp, status, binance_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.coin, "BUY", order_params["quantity"], 
                    order_params["price"], dt.datetime.utcnow().isoformat(),
                    "PLACED", order["orderId"]
                ))
                conn.commit()
                conn.close()
            except BinanceAPIException as e:
                logger.error(f"Binance error placing order: {e}")
            except Exception as e:
                logger.error(f"Error placing order for {self.coin}: {e}")
        
        return placed > 0

    def check_and_sell(self) -> bool:
        """Check if price reached profit target and sell."""
        current_price = self.get_current_price()
        if not current_price:
            return False
        
        # Get current position from DB
        conn = sqlite3.connect(CRYPTO_DB_PATH)
        cur = conn.cursor()
        cur.execute("SELECT * FROM crypto_positions WHERE coin = ?", (self.coin,))
        pos = cur.fetchone()
        conn.close()
        
        if not pos:
            return False
        
        entry_price = pos[2]
        profit_target = entry_price * (1 + self.profit_percent / 100)
        
        if current_price >= profit_target:
            # Execute sell
            try:
                order = self.client.order_market_sell(
                    symbol=self.symbol,
                    quantity=pos[1]
                )
                logger.info(f"Sold {self.coin}: {order['orderId']}")
                
                # Update DB
                conn = sqlite3.connect(CRYPTO_DB_PATH)
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO crypto_grid_orders
                    (coin, side, quantity, price, timestamp, status, binance_order_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.coin, "SELL", pos[1], current_price,
                    dt.datetime.utcnow().isoformat(), "FILLED", order["orderId"]
                ))
                cur.execute("DELETE FROM crypto_positions WHERE coin = ?", (self.coin,))
                conn.commit()
                conn.close()
                return True
            except Exception as e:
                logger.error(f"Error selling {self.coin}: {e}")
                return False
        
        return False

# ---------------------------
# 24/7 CRYPTO LOOP
# ---------------------------
def crypto_trading_loop():
    """Background loop for 24/7 crypto grid trading."""
    while True:
        client = get_binance_client()
        if not client:
            st.session_state["crypto_status"] = "Waiting for Binance API keys..."
            time.sleep(60)
            continue
        
        st.session_state["crypto_status"] = "Running grid trading..."
        
        for coin in GRID_COINS:
            try:
                params = GRID_TRADING_PARAMS[coin]
                trader = GridTrader(client, coin, params)
                
                # Check if position exists, if not create
                conn = sqlite3.connect(CRYPTO_DB_PATH)
                cur = conn.cursor()
                cur.execute("SELECT * FROM crypto_positions WHERE coin = ?", (coin,))
                pos = cur.fetchone()
                conn.close()
                
                if not pos:
                    trader.place_grid_orders()
                else:
                    trader.check_and_sell()
            except Exception as e:
                logger.error(f"Error in grid trading for {coin}: {e}")
        
        st.session_state["crypto_status"] = "Grid trading cycle complete. Running 24/7..."
        time.sleep(60)  # Run every minute

# ---------------------------
# STATE INITIALIZATION
# ---------------------------
def init_crypto_state():
    if "crypto_status" not in st.session_state:
        st.session_state["crypto_status"] = "Idle"
    if "crypto_running" not in st.session_state:
        st.session_state["crypto_running"] = False
    if "crypto_loop_started" not in st.session_state:
        st.session_state["crypto_loop_started"] = False
    if "crypto_positions" not in st.session_state:
        st.session_state["crypto_positions"] = {}

# ---------------------------
# UI FUNCTIONS
# ---------------------------
def get_crypto_positions() -> pd.DataFrame:
    """Load crypto positions from DB."""
    try:
        conn = sqlite3.connect(CRYPTO_DB_PATH)
        df = pd.read_sql("SELECT coin, quantity, entry_price, current_price FROM crypto_positions", conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()

def get_crypto_trades() -> pd.DataFrame:
    """Load recent crypto trades from DB."""
    try:
        conn = sqlite3.connect(CRYPTO_DB_PATH)
        df = pd.read_sql("""
            SELECT coin, side, quantity, price, timestamp FROM crypto_grid_orders 
            ORDER BY timestamp DESC LIMIT 100
        """, conn)
        conn.close()
        return df
    except:
        return pd.DataFrame()
