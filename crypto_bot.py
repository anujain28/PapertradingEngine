import pandas as pd
import time
import streamlit as st
import datetime as dt
import random
import yfinance as yf

# Initialize session state for the bot
if "crypto_positions" not in st.session_state:
    st.session_state["crypto_positions"] = pd.DataFrame(columns=["symbol", "entry_price", "pnl", "timestamp"])

def init_crypto_state():
    if "crypto_status" not in st.session_state:
        st.session_state["crypto_status"] = "Idle"
    if "crypto_running" not in st.session_state:
        st.session_state["crypto_running"] = False

def get_crypto_positions():
    return st.session_state.get("crypto_positions", pd.DataFrame())

def fetch_live_price(symbol):
    """Fetches real-time price from YFinance"""
    try:
        data = yf.Ticker(symbol).history(period="1d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except:
        pass
    return None

def crypto_trading_loop():
    """
    Background loop that paper trades using LIVE market data.
    """
    # Assets to trade
    target_coins = ["BTC-USD", "ETH-USD", "SOL-USD"]
    
    while True:
        if st.session_state.get("crypto_running", False):
            st.session_state["crypto_status"] = "Scanning Live Market Prices..."
            
            # 1. Pick a random coin to analyze
            coin = random.choice(target_coins)
            
            # 2. Fetch REAL Live Price
            price = fetch_live_price(coin)
            
            if price:
                current_df = st.session_state.get("crypto_positions", pd.DataFrame())
                
                # SIMULATION RULE: If we don't have this coin, BUY it.
                # In a real bot, you would check RSI/MACD here.
                already_owned = False
                if not current_df.empty and "symbol" in current_df.columns:
                    if coin in current_df["symbol"].values:
                        already_owned = True
                
                if not already_owned:
                    new_trade = pd.DataFrame([{
                        "symbol": coin, 
                        "entry_price": float(f"{price:.2f}"), 
                        "pnl": 0.0, # Starts at 0
                        "timestamp": dt.datetime.now().strftime("%H:%M:%S")
                    }])
                    
                    st.session_state["crypto_positions"] = pd.concat([current_df, new_trade], ignore_index=True)
                    st.session_state["crypto_status"] = f"Bought {coin} at ${price:.2f}"
            
            # Sleep to simulate analysis time and prevent API spam
            time.sleep(10)
            
        else:
            time.sleep(2)
