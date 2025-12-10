import streamlit as st
import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx

# Import Local Modules
try:
    from utils import apply_custom_style, init_db
    import crypto
    import stocks
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    ERR_MSG = str(e)

st.set_page_config(page_title="Paisa Banao Engine", layout="wide", page_icon="💰")

def main():
    if not MODULES_LOADED:
        st.error(f"Modules missing! Ensure utils.py, crypto.py, stocks.py are in the folder. Error: {ERR_MSG}")
        return

    apply_custom_style()
    init_db()
    crypto.init_crypto_state()
    stocks.init_stock_state()

    # --- SIDEBAR ---
    st.sidebar.title("💰 Paisa Banao")
    
    st.sidebar.subheader("📈 Stocks Menu")
    page_stocks = st.sidebar.radio("Stocks Actions", ["Bomb Stocks", "PNL Log"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🪙 Crypto Menu")
    page_crypto = st.sidebar.radio("Crypto Actions", ["Crypto Auto Pilot", "Crypto Report", "Manual Bot"], label_visibility="collapsed")
    
    st.sidebar.markdown("---")
    with st.sidebar.expander("🔑 Config & Keys"):
        t_tok = st.text_input("Telegram Token", type="password")
        t_id = st.text_input("Telegram Chat ID")
        if st.button("Save Telegram"):
            st.session_state["tg_token"] = t_tok
            st.session_state["tg_chat_id"] = t_id
    
    # --- THREADS ---
    if not st.session_state.get("loop_started", False):
        st.session_state["loop_started"] = True
        t = threading.Thread(target=crypto.crypto_trading_loop, daemon=True)
        add_script_run_ctx(t)
        t.start()

    # --- ROUTING ---
    # Determine active section
    if "last_stock" not in st.session_state: st.session_state["last_stock"] = page_stocks
    if "last_crypto" not in st.session_state: st.session_state["last_crypto"] = page_crypto
    if "mode" not in st.session_state: st.session_state["mode"] = "crypto"

    if page_stocks != st.session_state["last_stock"]:
        st.session_state["mode"] = "stocks"
        st.session_state["last_stock"] = page_stocks
    elif page_crypto != st.session_state["last_crypto"]:
        st.session_state["mode"] = "crypto"
        st.session_state["last_crypto"] = page_crypto

    # Render
    if st.session_state["mode"] == "stocks":
        if page_stocks == "Bomb Stocks": stocks.run_stocks_app()
        else: st.info("Stocks PNL Log Coming Soon")
    else:
        if page_crypto == "Crypto Auto Pilot": crypto.show_ai_autopilot_page()
        elif page_crypto == "Crypto Report": crypto.show_crypto_report_page()
        elif page_crypto == "Manual Bot": crypto.show_crypto_manual_bot_page()

if __name__ == "__main__":
    main()
