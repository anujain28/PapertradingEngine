import streamlit as st
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Import Modules
try:
    import app1 # Stocks
    from utils import apply_custom_style, init_db
    from crypto import (
        init_crypto_state, show_crypto_manual_bot_page, 
        show_ai_autopilot_page, show_crypto_report_page, 
        crypto_trading_loop
    )
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    ERR_MSG = str(e)

# --- PAGE CONFIG ---
st.set_page_config(page_title="Paisa Banao Engine", layout="wide", page_icon="💰")

def main():
    if not MODULES_LOADED:
        st.error(f"Modules missing! Ensure 'utils.py', 'crypto.py', and 'app1.py' are in the same folder. Error: {ERR_MSG}")
        return

    # Init
    apply_custom_style()
    init_db()
    init_crypto_state()
    app1.init_stock_state()

    # --- SIDEBAR ---
    st.sidebar.title("💰 Paisa Banao")
    st.sidebar.title("Navigation")
    
    st.sidebar.subheader("📈 Stocks Menu")
    page_stocks = st.sidebar.radio("Stocks Actions", ["Bomb Stocks", "PNL Log", "Auto-Pilot App"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🪙 Crypto Menu")
    page_crypto = st.sidebar.radio("Crypto Actions", ["Crypto Auto Pilot", "Crypto Report", "Manual Bot"], label_visibility="collapsed")
    
    # Configs
    st.sidebar.markdown("---")
    with st.sidebar.expander("📢 Telegram Alerts"):
        t1 = st.text_input("Bot Token", value=st.session_state.get("tg_token", ""), type="password")
        t2 = st.text_input("Chat ID", value=st.session_state.get("tg_chat_id", ""))
        if st.button("💾 Save Telegram"):
            st.session_state["tg_token"] = t1; st.session_state["tg_chat_id"] = t2; st.success("Saved!")

    with st.sidebar.expander("🔌 Binance Keys"):
        k1 = st.text_input("API Key", type="password")
        k2 = st.text_input("Secret Key", type="password")
        if st.button("💾 Save Binance"): st.success("Saved!")

    with st.sidebar.expander("🇮🇳 Dhan Config"):
        d1 = st.text_input("Client ID")
        d2 = st.text_input("Access Token", type="password")
        if st.button("💾 Save Dhan"): st.success("Saved!")

    # --- NAVIGATION LOGIC ---
    if "last_stock" not in st.session_state: st.session_state["last_stock"] = page_stocks
    if "last_crypto" not in st.session_state: st.session_state["last_crypto"] = page_crypto
    if "active_sec" not in st.session_state: st.session_state["active_sec"] = "crypto"
    
    current_page = "Crypto Auto Pilot"
    
    if page_stocks != st.session_state["last_stock"]:
        current_page = page_stocks
        st.session_state["active_sec"] = "stocks"
        st.session_state["last_stock"] = page_stocks
    elif page_crypto != st.session_state["last_crypto"]:
        current_page = page_crypto
        st.session_state["active_sec"] = "crypto"
        st.session_state["last_crypto"] = page_crypto
    else:
        if st.session_state["active_sec"] == "stocks": current_page = page_stocks
        else: current_page = page_crypto

    # --- BACKGROUND THREADS ---
    if not st.session_state.get("loop_started", False):
        st.session_state["loop_started"] = True
        t = threading.Thread(target=crypto_trading_loop, daemon=True)
        add_script_run_ctx(t)
        t.start()

    # --- ROUTING ---
    if current_page == "Crypto Auto Pilot": show_ai_autopilot_page()
    elif current_page == "Crypto Report": show_crypto_report_page()
    elif current_page == "Manual Bot": show_crypto_manual_bot_page()
    elif current_page == "Auto-Pilot App": app1.show_stock_autopilot()
    elif current_page == "Bomb Stocks": app1.show_stocks_dashboard()
    elif current_page == "PNL Log": st.title("Stocks PNL (Coming Soon)")

if __name__ == "__main__":
    main()
