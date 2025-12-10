# Complete Stock Analysis Bot - GROWW + DHAN + CONFIG PAGES MODIFIED
# All other logic preserved

import subprocess, sys

def ensure_package(pkg_name: str):
    try:
        __import__(pkg_name)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])

try:
    ensure_package("dhanhq")
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional
import pytz
import requests
import os
import json
from pathlib import Path
from streamlit_local_storage import LocalStorage

try:
    from dhanhq import dhanhq
except ImportError:
    dhanhq = None

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {"dhan_client_id": "", "telegram_bot_token": "", "telegram_chat_id": "", "notify_enabled": False}

def load_config():
    if not os.path.exists(CONFIG_FILE):
        return DEFAULT_CONFIG.copy()
    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        cfg = DEFAULT_CONFIG.copy()
        cfg.update(data)
        return cfg
    except Exception:
        return DEFAULT_CONFIG.copy()

def save_config_from_state():
    cfg = {
        "dhan_client_id": st.session_state.get("dhan_client_id", ""),
        "telegram_bot_token": st.session_state.get("telegram_bot_token", ""),
        "telegram_chat_id": st.session_state.get("telegram_chat_id", ""),
        "notify_enabled": st.session_state.get("notify_enabled", False),
    }
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        st.warning(f"Could not save config: {e}")

st.set_page_config(page_title="🤖 AI Stock Analysis Bot", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .stApp { background-color: #fafafa; color: #111827; }
    body { background-color: #fafafa; color: #111827; }

    .main-header {
        background: linear-gradient(120deg, #4f46e5 0%, #0ea5e9 100%);
        padding: 18px; border-radius: 18px; color: white; margin-bottom: 12px;
        box-shadow: 0 12px 28px rgba(15,23,42,0.35); border: 1px solid rgba(255,255,255,0.14);
    }
    .main-header h1 { margin-bottom: 4px; font-size: clamp(1.6rem, 3vw, 2.3rem); }
    .main-header p { margin: 0; font-size: 0.9rem; opacity: 0.96; }
    .status-badge {
        display: inline-block; padding: 4px 10px; border-radius: 999px; font-size: 0.7rem;
        text-transform: uppercase; letter-spacing: 0.07em; background: rgba(15,23,42,0.35);
        border: 1px solid rgba(226,232,240,0.7); margin-top: 6px;
    }
    .metric-card {
        padding: 12px; border-radius: 14px; background: #ffffff; border: 1px solid #e5e7eb;
        box-shadow: 0 4px 12px rgba(15,23,42,0.18); margin-bottom: 10px; color: #111827;
    }
    .metric-card h3 { font-size: 0.95rem; color: #0f172a; margin-bottom: 4px; }
    .metric-card .value { font-size: 1.05rem; font-weight: 600; color: #111827; }
    .metric-card .sub { font-size: 0.8rem; color: #4b5563; }
    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
    .chip { padding: 2px 8px; border-radius: 999px; font-size: 0.7rem; background: #ffffff; border: 1px solid #d1d5db; color: #111827; }
    .st-key-refresh_btn button { background-color: #f97316 !important; color: #ffffff !important; border-color: #ea580c !important; }
    .st-key-refresh_btn button:hover { background-color: #ea580c !important; }

    /* File uploader dark button with light font */
    div[data-testid="stFileUploader"] section button {
        background-color: #1e293b !important; color: #f8fafc !important;
        border: 1px solid #334155 !important; font-weight: 500 !important;
    }
    div[data-testid="stFileUploader"] section button:hover {
        background-color: #334155 !important; color: #ffffff !important;
    }

    .st-key-btn_auto_calc button {
        background-color: #1e293b !important; color: #f8fafc !important;
        border-color: #334155 !important; font-weight: 500 !important;
    }
    .st-key-btn_auto_calc button:hover {
        background-color: #334155 !important; color: #ffffff !important;
    }

    /* Telegram config buttons – make clearly visible */
    .st-key-btn_save_settings button,
    .st-key-btn_send_pnl button {
        background-color: #0ea5e9 !important;
        color: #0f172a !important;
        border-color: #38bdf8 !important;
        font-weight: 600 !important;
    }
    .st-key-btn_save_settings button:hover,
    .st-key-btn_send_pnl button:hover {
        background-color: #0284c7 !important;
        color: #e5f3ff !important;
    }

    /* 🔳 Dark table for portfolio snapshot, projections, and configs */
    .dark-table {
        width: 100%;
        border-collapse: collapse;
        background-color: #020617;
        color: #f9fafb;
        border-radius: 12px;
        overflow: hidden;
        margin-top: 8px;
        margin-bottom: 12px;
    }
    .dark-table th, .dark-table td {
        padding: 8px 10px;
        border: 1px solid #1f2937;
        font-size: 0.85rem;
    }
    .dark-table th {
        background-color: #111827;
        font-weight: 600;
        text-align: left;
    }

    /* Make ALL Streamlit tables/dataframes dark-looking */
    div[data-testid="stDataFrame"] table {
        background-color: #020617 !important;
        color: #f9fafb !important;
    }
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td {
        background-color: #020617 !important;
        color: #f9fafb !important;
        border-color: #1f2937 !important;
        font-size: 0.85rem !important;
    }
</style>
""", unsafe_allow_html=True)

IST = pytz.timezone('Asia/Kolkata')
_cfg = load_config()
localS = LocalStorage()

# Session state
for key, default in [
    ('last_analysis_time', None),
    ('last_auto_scan', None),
    ('recommendations', {'BTST': [], 'Intraday': [], 'Weekly': [], 'Monthly': []}),
    ('current_page', "🔥 Top Stocks"),
    ('dhan_enabled', False),
    ('dhan_client_id', _cfg.get('dhan_client_id', '')),
    ('dhan_access_token', ''),
    ('dhan_client', None),
    ('dhan_login_msg', 'Not configured'),
    ('notify_enabled', _cfg.get('notify_enabled', False)),
    ('telegram_bot_token', _cfg.get('telegram_bot_token', '')),
    ('telegram_chat_id', _cfg.get('telegram_chat_id', '')),
    ('last_pnl_notify', None),
    ('last_reco_notify', {}),   # track last recommendation sends per slot
]:
    if key not in st.session_state:
        st.session_state[key] = default

DATA_DIR = Path(__file__).parent / "data"
NIFTY200_CSV = DATA_DIR / "nifty200_yahoo.csv"
MASTER_NIFTY200 = DATA_DIR / "ind_nifty200list.csv"

@st.cache_data
def load_nifty200_universe():
    if not NIFTY200_CSV.exists():
        st.error(f"Universe file not found: {NIFTY200_CSV}")
        return [], {}
    df = pd.read_csv(NIFTY200_CSV)
    if "SYMBOL" not in df.columns or "YF_TICKER" not in df.columns:
        st.error("nifty200_yahoo.csv must have columns: SYMBOL, YF_TICKER")
        return [], {}
    df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
    df["YF_TICKER"] = df["YF_TICKER"].astype(str).str.strip()
    symbols = df["SYMBOL"].dropna().unique().tolist()
    mapping = dict(zip(df["SYMBOL"], df["YF_TICKER"]))
    return symbols, mapping

STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()

def regenerate_nifty200_csv_from_master():
    if not MASTER_NIFTY200.exists():
        st.error(f"Master list not found: {MASTER_NIFTY200}")
        return False
    df_src = pd.read_csv(MASTER_NIFTY200)
    if "Symbol" not in df_src.columns:
        st.error("ind_nifty200list.csv must have a 'Symbol' column")
        return False
    df_out = pd.DataFrame()
    df_out["SYMBOL"] = df_src["Symbol"].astype(str).str.strip().str.upper()
    df_out["YF_TICKER"] = df_out["SYMBOL"].apply(lambda s: f"{s}.NS")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(NIFTY200_CSV, index=False)
    # Clear cache of the loader if supported
    try:
        st.cache_data.clear()
    except Exception:
        try:
            load_nifty200_universe.clear()
        except Exception:
            pass
    global STOCK_UNIVERSE, NIFTY_YF_MAP
    STOCK_UNIVERSE, NIFTY_YF_MAP = load_nifty200_universe()
    return True

def nse_yf_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = str(sym).strip().upper()
    # remove common punctuation
    s = s.replace(".", "").replace("/", "").split()[0]
    if s in NIFTY_YF_MAP:
        return NIFTY_YF_MAP[s]
    return s if s.endswith(".NS") else f"{s}.NS"

def dhan_login(client_id: str, access_token: str):
    if not dhanhq:
        st.session_state['dhan_client'] = None
        st.session_state['dhan_login_msg'] = "dhanhq not installed"
        return
    try:
        d = dhanhq(client_id, access_token)
        _ = d.get_holdings()
        st.session_state['dhan_client'] = d
        st.session_state['dhan_login_msg'] = "✅ Dhan client configured"
    except Exception as e:
        st.session_state['dhan_client'] = None
        st.session_state['dhan_login_msg'] = f"❌ Dhan error: {e}"

def dhan_logout():
    st.session_state['dhan_client'] = None
    st.session_state['dhan_login_msg'] = "Logged out"

def get_dhan_raw():
    d = st.session_state.get('dhan_client')
    if not d:
        return pd.DataFrame(), pd.DataFrame()
    try:
        h = d.get_holdings()
        holdings = h.get("data", []) if isinstance(h, dict) else h
    except Exception:
        holdings = []
    try:
        p = d.get_positions()
        positions = p.get("data", []) if isinstance(p, dict) else p
    except Exception:
        positions = []
    return pd.DataFrame(holdings), pd.DataFrame(positions)

def format_dhan_portfolio_table():
    h_df, p_df = get_dhan_raw()
    if h_df.empty and p_df.empty:
        return pd.DataFrame(), 0.0
    if not h_df.empty:
        name_col = next((c for c in ['securityName', 'tradingSymbol', 'symbol'] if c in h_df.columns), None)
        qty_col = next((c for c in ['quantity', 'netQty', 'buyQty', 'totalQty'] if c in h_df.columns), None)
        avg_col = next((c for c in ['averagePrice', 'buyAvg', 'avgCostPrice'] if c in h_df.columns), None)
        cmp_col = next((c for c in ['ltp', 'lastTradedPrice', 'lastPrice'] if c in h_df.columns), None)
        h_df['_name'] = h_df[name_col] if name_col else ""
        h_df['_qty'] = pd.to_numeric(h_df[qty_col], errors='coerce').fillna(0.0) if qty_col else 0.0
        h_df['_avg'] = pd.to_numeric(h_df[avg_col], errors='coerce').fillna(0.0) if avg_col else 0.0
        h_df['_cmp'] = pd.to_numeric(h_df[cmp_col], errors='coerce') if cmp_col else np.nan
        h_df['_total_cost'] = h_df['_qty'] * h_df['_avg']
        h_df['_total_price'] = h_df['_qty'] * h_df['_cmp']
        h_df['_pnl'] = h_df['_total_price'] - h_df['_total_cost']
        portfolio = h_df[['_name', '_qty', '_avg', '_total_cost', '_cmp', '_total_price', '_pnl']].rename(
            columns={'_name': 'Stock', '_qty': 'Quantity', '_avg': 'Avg Cost', '_total_cost': 'Total Cost',
                     '_cmp': 'CMP', '_total_price': 'Total Value', '_pnl': 'P&L'}
        )
    else:
        name_col = next((c for c in ['tradingSymbol', 'securityName', 'symbol'] if c in p_df.columns), None)
        p_df['_name'] = p_df[name_col] if name_col else ""
        p_df['_qty'] = pd.to_numeric(p_df['netQty'], errors='coerce').fillna(0.0)
        p_df['_avg'] = pd.to_numeric(p_df['avgPrice'], errors='coerce').fillna(0.0)
        p_df['_cmp'] = pd.to_numeric(p_df['ltp'], errors='coerce').fillna(0.0)
        p_df['_total_cost'] = p_df['_qty'] * p_df['_avg']
        p_df['_total_price'] = p_df['_qty'] * p_df['_cmp']
        p_df['_pnl'] = p_df['_total_price'] - p_df['_total_cost']
        portfolio = p_df[['_name', '_qty', '_avg', '_total_cost', '_cmp', '_total_price', '_pnl']].rename(
            columns={'_name': 'Stock', '_qty': 'Quantity', '_avg': 'Avg Cost', '_total_cost': 'Total Cost',
                     '_cmp': 'CMP', '_total_price': 'Total Value', '_pnl': 'P&L'}
        )
    total_pnl = float(portfolio['P&L'].fillna(0).sum())
    return portfolio, total_pnl

def send_telegram_message(text: str):
    token = st.session_state.get('telegram_bot_token', '')
    chat_id = st.session_state.get('telegram_chat_id', '')
    if not token or not chat_id:
        return {"ok": False, "error": "Missing Telegram config"}
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        r = requests.get(url, params={"chat_id": chat_id, "text": text})
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

def safe_extract(df, col):
    if df is None or df.empty or col not in df.columns:
        return pd.Series(dtype=float)
    s = df[col]
    if isinstance(s, pd.DataFrame):
        s = s.squeeze()
    return pd.Series(s) if not isinstance(s, pd.Series) else s

def safe_scalar(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (pd.Series, list, np.ndarray)):
            x = np.ravel(x)[-1]
        return float(x)
    except Exception:
        return np.nan

class TechnicalAnalysis:
    @staticmethod
    def rsi_signal(df):
        c = safe_extract(df, 'Close')
        if len(c) < 15:
            return False, None, 0
        rsi = ta.momentum.RSIIndicator(c, window=14).rsi()
        v = safe_scalar(rsi.iloc[-1])
        if v < 30:
            return True, f"RSI Oversold ({v:.1f})", 35
        if v < 40:
            return True, f"RSI Strong Buy Zone ({v:.1f})", 25
        return False, None, 0

    @staticmethod
    def macd_signal(df):
        c = safe_extract(df, 'Close')
        if len(c) < 35:
            return False, None, 0
        macd = ta.trend.MACD(c)
        m = safe_scalar(macd.macd().iloc[-1])
        s = safe_scalar(macd.macd_signal().iloc[-1])
        h = safe_scalar(macd.macd_diff().iloc[-1])
        if m > s and h > 0:
            return True, "MACD Bullish Crossover", 30
        return False, None, 0

    @staticmethod
    def stochastic_signal(df):
        h, l, c = safe_extract(df, 'High'), safe_extract(df, 'Low'), safe_extract(df, 'Close')
        if len(c) < 14:
            return False, None, 0
        stoch = ta.momentum.StochasticOscillator(h, l, c, window=14, smooth_window=3)
        k = safe_scalar(stoch.stoch().iloc[-1])
        d = safe_scalar(stoch.stoch_signal().iloc[-1])
        if k < 20 and k > d:
            return True, "Stochastic Oversold Reversal", 25
        return False, None, 0

    @staticmethod
    def adx_trend_strength(df):
        h, l, c = safe_extract(df, 'High'), safe_extract(df, 'Low'), safe_extract(df, 'Close')
        if len(c) < 25:
            return False, None, 0
        adx = ta.trend.ADXIndicator(h, l, c, window=14)
        adx_val = safe_scalar(adx.adx().iloc[-1])
        plus_di = safe_scalar(adx.adx_pos().iloc[-1])
        minus_di = safe_scalar(adx.adx_neg().iloc[-1])
        if adx_val > 25 and plus_di > minus_di:
            return True, f"Strong Uptrend (ADX: {adx_val:.1f})", 30
        return False, None, 0

    @staticmethod
    def bollinger_squeeze(df):
        c = safe_extract(df, 'Close')
        if len(c) < 20:
            return False, None, 0
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        lower = safe_scalar(bb.bollinger_lband().iloc[-1])
        upper = safe_scalar(bb.bollinger_hband().iloc[-1])
        price = safe_scalar(c.iloc[-1])
        bandwidth = (upper - lower) / price * 100 if price and not np.isnan(price) else 0
        if price < lower * 1.02 and bandwidth < 10:
            return True, "BB Squeeze Breakout Setup", 25
        return False, None, 0

    @staticmethod
    def volume_price_confirmation(df):
        v = safe_extract(df, 'Volume')
        c = safe_extract(df, 'Close')
        if len(v) < 20:
            return False, None, 0
        avg_vol = v.rolling(20).mean().iloc[-1]
        cur_vol = v.iloc[-1]
        # guards for indexing
        if len(c) < 2 or pd.isna(c.iloc[-1]) or pd.isna(c.iloc[-2]):
            return False, None, 0
        chg = ((c.iloc[-1] - c.iloc[-2]) / c.iloc[-2]) * 100
        if cur_vol > avg_vol * 1.5 and chg > 1:
            return True, f"Volume Spike with Price Up ({chg:.1f}%)", 30
        return False, None, 0

    @staticmethod
    def ema_crossover(df):
        c = safe_extract(df, 'Close')
        if len(c) < 50:
            return False, None, 0
        ema9 = ta.trend.EMAIndicator(c, window=9).ema_indicator()
        ema21 = ta.trend.EMAIndicator(c, window=21).ema_indicator()
        if safe_scalar(ema9.iloc[-1]) > safe_scalar(ema21.iloc[-1]) and safe_scalar(ema9.iloc[-2]) <= safe_scalar(ema21.iloc[-2]):
            return True, "EMA 9/21 Golden Cross", 28
        return False, None, 0

    @staticmethod
    def obv_divergence(df):
        c = safe_extract(df, 'Close')
        v = safe_extract(df, 'Volume')
        if len(c) < 20:
            return False, None, 0
        obv = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
        obv_sma = obv.rolling(10).mean()
        if safe_scalar(obv.iloc[-1]) > safe_scalar(obv_sma.iloc[-1]):
            return True, "OBV Accumulation Phase", 22
        return False, None, 0

    @staticmethod
    def calculate_targets(df, price):
        h, l, c = safe_extract(df, 'High'), safe_extract(df, 'Low'), safe_extract(df, 'Close')
        if len(c) < 14:
            return {}
        atr = ta.volatility.AverageTrueRange(h, l, c, window=14).average_true_range()
        atr_v = safe_scalar(atr.iloc[-1])
        return {
            'atr': float(round(atr_v, 2)),
            'stop_loss': float(round(price - (atr_v * 2), 2)),
            'target_1': float(round(price + (atr_v * 2), 2)),
            'target_2': float(round(price + (atr_v * 3), 2)),
            'target_3': float(round(price + (atr_v * 4), 2)),
            'risk_reward': '1:2-4'
        }

def analyze_stock(ticker: str, period_type: str) -> Optional[Dict]:
    cfgs = {
        'BTST': {'period': '10d', 'interval': '15m'},
        'Intraday': {'period': '5d', 'interval': '15m'},
        'Weekly': {'period': '90d', 'interval': '1d'},
        'Monthly': {'period': '1y', 'interval': '1d'}
    }
    cfg = cfgs[period_type]
    try:
        t = yf.Ticker(nse_yf_symbol(ticker))
        df = t.history(period=cfg['period'], interval=cfg['interval'], auto_adjust=True)
        if df is None or df.empty or len(df) < 30:
            return None
        df = df.reset_index()
        df.columns = [str(c).capitalize() for c in df.columns]
        strategies = [
            (TechnicalAnalysis.rsi_signal, 'RSI'),
            (TechnicalAnalysis.macd_signal, 'MACD'),
            (TechnicalAnalysis.stochastic_signal, 'Stochastic'),
            (TechnicalAnalysis.adx_trend_strength, 'ADX'),
            (TechnicalAnalysis.bollinger_squeeze, 'Bollinger'),
            (TechnicalAnalysis.volume_price_confirmation, 'Volume'),
            (TechnicalAnalysis.ema_crossover, 'EMA'),
            (TechnicalAnalysis.obv_divergence, 'OBV')
        ]
        signals, reasons, score = [], [], 0
        for func, name in strategies:
            sig, reason, s_ = func(df)
            if sig:
                signals.append(name)
                if reason:
                    reasons.append(reason)
                score += s_
        if len(signals) < 3:
            return None
        price = float(df['Close'].iloc[-1])
        targets = TechnicalAnalysis.calculate_targets(df, price)
        strength = "STRONG BUY" if score >= 90 else "BUY" if score >= 60 else "HOLD"
        timeframe_map = {
            'BTST': '1–3 days (BTST)',
            'Intraday': 'Same day',
            'Weekly': 'Up to 1 week',
            'Monthly': '2–4 weeks'
        }
        timeframe = timeframe_map.get(period_type, '1–5 days')
        return {
            'ticker': ticker,
            'price': round(price, 2),
            'signals_count': len(signals),
            'strategies': ", ".join(signals),
            'score': score,
            'reasons': " | ".join(reasons),
            'signal_strength': strength,
            'period': period_type,
            'timeframe': timeframe,
            **targets
        }
    except Exception:
        return None

def analyze_multiple_stocks(tickers: List[str], period_type: str, max_results: int = 10) -> List[Dict]:
    out = []
    if not tickers:
        return out
    bar = st.progress(0)
    txt = st.empty()
    for i, tck in enumerate(tickers):
        txt.text(f"🔍 Analyzing {period_type}: {tck} ({i+1}/{len(tickers)})")
        bar.progress((i + 1) / len(tickers))
        res = analyze_stock(tck, period_type)
        if res:
            out.append(res)
    try:
        bar.empty()
    except Exception:
        pass
    txt.empty()
    return sorted(out, key=lambda x: x['score'], reverse=True)[:max_results]

def run_analysis():
    now = datetime.now(IST)
    st.session_state['last_analysis_time'] = now
    st.session_state['last_auto_scan'] = now
    for p in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        st.session_state['recommendations'][p] = analyze_multiple_stocks(STOCK_UNIVERSE, p, max_results=20)

def market_hours_window(dt: datetime):
    start = dt.replace(hour=9, minute=10, second=0, microsecond=0)
    end = dt.replace(hour=15, minute=40, second=0, microsecond=0)
    return start <= dt <= end

def fmt_lakhs(x: float) -> str:
    try:
        return f"₹{x/1e5:.2f} L"
    except Exception:
        return "₹0.00 L"

def project_value(current_value: float, cagr: float, yearly_dividend: float, years: int) -> float:
    future = current_value * ((1 + cagr) ** years)
    future += yearly_dividend * years
    return float(future)

# ---------- TELEGRAM RECOMMENDATION SCHEDULER ----------

NOTIFY_SLOTS = ["09:30", "13:30", "15:00"]  # IST slots

def build_telegram_reco_message(now: datetime) -> str:
    lines = []
    lines.append("📊 Auto Stock Ideas (NIFTY 200)")
    lines.append(f"🕒 {now.strftime('%d-%m-%Y %H:%M')} IST")
    lines.append("")

    recs_state = st.session_state.get("recommendations", {})
    for period in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        recs = recs_state.get(period, [])
        if not recs:
            continue
        tmp = sorted(recs, key=lambda x: x.get("score", 0), reverse=True)[:3]
        lines.append(f"• {period}:")
        for r in tmp:
            tck = r.get("ticker", "")
            price = r.get("price", 0.0)
            strength = r.get("signal_strength", "")
            tgt = r.get("target_1", None)
            if tgt is not None:
                try:
                    tgt_fmt = f"{float(tgt):.2f}"
                except Exception:
                    tgt_fmt = str(tgt)
                lines.append(f"   - {tck}: ₹{price:.2f} → 🎯 ₹{tgt_fmt} ({strength})")
            else:
                lines.append(f"   - {tck}: ₹{price:.2f} ({strength})")
        lines.append("")
    if len(lines) <= 3:
        lines.append("No strong signals available right now.")
    lines.append("#AutoScan #NSE #Nifty200")
    return "\n".join(lines)

def send_scheduled_recommendations(now: datetime):
    if not st.session_state.get("notify_enabled", False):
        return
    recs = st.session_state.get("recommendations", {})
    if not any(recs.get(k) for k in ['BTST', 'Intraday', 'Weekly', 'Monthly']):
        run_analysis()
    msg = build_telegram_reco_message(now)
    send_telegram_message(msg)
    st.caption("📤 Telegram recommendations sent.")
    st.session_state['last_pnl_notify'] = now

def handle_scheduled_notifications(now: datetime):
    if not st.session_state.get("notify_enabled", False):
        return

    today_str = now.strftime("%Y-%m-%d")
    last_map = st.session_state.get("last_reco_notify", {}) or {}

    for slot in NOTIFY_SLOTS:
        hr, mn = map(int, slot.split(":"))
        scheduled_dt = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
        window_start = scheduled_dt
        window_end = scheduled_dt + timedelta(minutes=5)
        already_sent_today = last_map.get(slot) == today_str

        if (now >= window_start) and (now <= window_end) and not already_sent_today:
            send_scheduled_recommendations(now)
            last_map[slot] = today_str
            st.session_state['last_reco_notify'] = last_map

def auto_scan_if_due():
    now = datetime.now(IST)
    last = st.session_state.get('last_auto_scan')
    if market_hours_window(now):
        should_run = False
        if last is None:
            should_run = True
        else:
            try:
                if (now - last).total_seconds() >= 20 * 60:
                    should_run = True
            except Exception:
                should_run = True
        if should_run:
            run_analysis()
            st.caption(f"🕒 Auto-scan executed at {now.strftime('%H:%M:%S')} IST")
    handle_scheduled_notifications(now)

def get_top_stocks(limit: int = 10):
    all_recs = []
    for period in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        for r in st.session_state['recommendations'].get(period, []):
            rec = dict(r)
            rec['period'] = period
            all_recs.append(rec)
    if not all_recs:
        return []
    df_all = pd.DataFrame(all_recs).sort_values("score", ascending=False)
    seen = set()
    unique_rows = []
    for _, row in df_all.iterrows():
        t = row.get('ticker')
        if t not in seen:
            seen.add(t)
            unique_rows.append(row)
        if len(unique_rows) >= limit:
            break
    if not unique_rows:
        return []
    return pd.DataFrame(unique_rows).to_dict(orient="records")

def load_groww_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Only CSV, XLS, or XLSX files are supported.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

def map_groww_columns(df: pd.DataFrame):
    norm_cols = {c.lower().strip(): c for c in df.columns}
    required_map = {
        "stock name": "stock name",
        "isin": "isin",
        "quantity": "quantity",
        "average buy price per share": "average buy price per share",
        "total investment": "total investment",
        "total cmp": "total cmp",
        "total p&l": "total p&l",
    }
    out, missing = {}, []
    for logical_key, norm_header in required_map.items():
        if norm_header in norm_cols:
            out[logical_key] = norm_cols[norm_header]
        else:
            missing.append(logical_key)
    if missing:
        msg = (
            "Columns must match this Groww template exactly: "
            "'Stock Name, ISIN, Quantity, Average buy price per share, "
            "Total Investment, Total CMP, TOTAL P&L'. "
            "Missing or mismatched: " + ", ".join(missing)
        )
        return None, msg
    return out, None

def fetch_dividend_and_cagr(stock_name: str, isin: str, cmp_value: float):
    sym = stock_name.split()[0].upper().strip() if stock_name else ""
    yf_ticker = NIFTY_YF_MAP.get(sym, None)
    if not yf_ticker and sym:
        yf_ticker = f"{sym}.NS"
    div_yield, div_rupees, cagr = 0.0, 0.0, 0.05
    if not yf_ticker:
        return div_yield, div_rupees, cagr
    try:
        t = yf.Ticker(yf_ticker)
        info = t.info or {}
        raw_yield = info.get("dividendYield")
        if raw_yield is not None:
            div_yield = float(raw_yield)
        if cmp_value and div_yield:
            div_rupees = div_yield * cmp_value
        hist = t.history(period="10y")
        if hist is not None and not hist.empty:
            hist = hist.dropna(subset=["Close"])
            first_price = float(hist["Close"].iloc[0])
            last_price = float(hist["Close"].iloc[-1])
            years = max((hist.index[-1] - hist.index[0]).days / 365.0, 1.0)
            if first_price > 0 and years > 0:
                cagr = (last_price / first_price) ** (1.0 / years) - 1.0
    except Exception:
        pass
    return float(div_yield), float(div_rupees), float(cagr)

def classify_strength(pct_pnl: float, cagr: float, price_zero: bool) -> str:
    if price_zero:
        return "Super Strong"
    if cagr >= 0.15 and pct_pnl >= 20:
        return "Super Strong"
    if cagr >= 0.10 and pct_pnl >= 0:
        return "Strong"
    if cagr >= 0.05 or pct_pnl > -10:
        return "Medium"
    if cagr >= 0.0 or pct_pnl > -30:
        return "Weak"
    return "Super Weak"

def get_recommendation(pct_pnl: float, cagr: float, price_zero: bool) -> str:
    strength = classify_strength(pct_pnl, cagr, price_zero)
    if strength in ["Super Strong", "Strong"]:
        return "BUY"
    elif strength == "Medium":
        return "HOLD"
    else:
        return "SELL"

def suggest_horizon(strength: str, div_yield: float, cagr: float) -> str:
    if strength == "Super Strong":
        if div_yield >= 0.015 or cagr >= 0.18:
            return "Hold 20+ years (core compounding)"
        return "Hold 15–20 years"
    if strength == "Strong":
        if cagr >= 0.12:
            return "Hold 10–15 years"
        return "Hold 7–10 years"
    if strength == "Medium":
        if cagr >= 0.08:
            return "Hold 5–7 years"
        return "Review in 3–5 years"
    if strength == "Weak":
        if cagr > 0:
            return "Tactical hold; reassess within 1–2 years"
        return "Exit gradually over 1–2 years"
    return "Exit within 6–12 months; rotate to better compounders"

def render_reco_cards(recs: List[Dict], label: str):
    if not recs:
        st.info(f"Tap 🚀 Run Full Scan to generate {label} ideas.")
        return
    df = pd.DataFrame(recs).sort_values("score", ascending=False).head(20 if label == "Top" else 10)
    for _, rec in df.iterrows():
        cmp_ = rec.get('price', 0.0)
        tgt = rec.get('target_1', np.nan)
        # safe formatting for target (avoid attempting {:.2f} on NaN)
        if tgt is None or (isinstance(tgt, float) and np.isnan(tgt)):
            tgt_display = "—"
            diff = np.nan
            profit_pct = np.nan
        else:
            try:
                tgt_val = float(tgt)
                tgt_display = f"{tgt_val:.2f}"
                diff = tgt_val - cmp_
                profit_pct = (diff / cmp_ * 100) if cmp_ else np.nan
            except Exception:
                tgt_display = str(tgt)
                diff = np.nan
                profit_pct = np.nan

        reason = rec.get('reasons', '')
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{rec.get('ticker','')} • {rec.get('signal_strength','')} ⚡</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>💰 CMP: ₹{cmp_:.2f}  | 🎯 Target: ₹{tgt_display}</div>", unsafe_allow_html=True)
        chip_html = "<div class='chip-row'>"
        chip_html += f"<span class='chip'>⭐ Score: {int(rec.get('score',0))}</span>"
        chip_html += f"<span class='chip'>⏱ {rec.get('timeframe','')}</span>"
        chip_html += f"<span class='chip'>📊 {rec.get('period',label)}</span>"
        chip_html += "</div>"
        st.markdown(chip_html, unsafe_allow_html=True)
        if not np.isnan(diff):
            st.markdown(f"<div class='sub'>📈 Target Profit: ₹{diff:.2f} • 💹 Profit %: {profit_pct:.2f}%</div>", unsafe_allow_html=True)
        if reason:
            st.markdown(f"<div class='sub'>🧠 Reason: {reason}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# 👉 MF Analysis tab added between Dhan and Configuration
NAV_PAGES = [
    "🔥 Top Stocks",
    "🌙 BTST",
    "⚡ Intraday",
    "📆 Weekly",
    "📅 Monthly",
    "📊 Groww",
    "🤝 Dhan",
    "🧾 Dhan Stocks Analysis",
    "📈 MF Analysis",
    "⚙️ Configuration",
]

def sidebar_nav():
    with st.sidebar:
        st.markdown("### 📂 Views")
        page = st.radio(
            "Navigation",
            NAV_PAGES,
            index=NAV_PAGES.index(st.session_state.get("current_page", "🔥 Top Stocks")),
            label_visibility="collapsed",
        )
        st.session_state["current_page"] = page

def analyze_dhan_portfolio_recommendations():
    """
    Fetch Dhan holdings and produce per-stock recommendations using AI heuristics:
    - uses YF history to compute ATR-based targets
    - compares CMP to target_1 and target_2 to output Buy/Hold/Sell and distance info
    - shows CMP, present P&L for that stock, upcoming dividend date & amount (if available),
      and labels recommendation with suggested timeframe (BTST/Intraday/Weekly/Monthly) from scanner
    """
    df_port, total_pnl = format_dhan_portfolio_table()
    if df_port is None or df_port.empty:
        st.info("No Dhan holdings/positions fetched yet. Connect Dhan to fetch holdings.")
        return

    st.markdown("### 🧾 Dhan Stocks — AI Recommendations")
    st.markdown("For each stock below: recommendation (Buy/Hold/Sell), target price, how far CMP is from recommended price, present P&L, and upcoming dividend (if any).")
    st.write("")  # spacing

    # iterate rows
    for _, row in df_port.iterrows():
        stock_name = str(row.get("Stock", "")).strip()
        # CMP, P&L and quantities from portfolio table
        cmp_val = float(row.get("CMP", 0.0)) if not pd.isna(row.get("CMP", np.nan)) else np.nan
        qty = float(row.get("Quantity", 0.0)) if not pd.isna(row.get("Quantity", np.nan)) else 0.0
        total_cost = float(row.get("Total Cost", 0.0)) if not pd.isna(row.get("Total Cost", np.nan)) else 0.0
        total_value = float(row.get("Total Value", 0.0)) if not pd.isna(row.get("Total Value", np.nan)) else 0.0
        pnl = float(row.get("P&L", 0.0)) if not pd.isna(row.get("P&L", np.nan)) else 0.0
        pct_pnl = (pnl / total_cost * 100.0) if total_cost else 0.0

        # Try to map to yahoo ticker
        yf_sym = nse_yf_symbol(stock_name)
        # Fetch recent history for targets & info
        targets = {}
        info = {}
        try:
            t = yf.Ticker(yf_sym)
            info = t.info or {}
            hist = t.history(period="90d", interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty:
                hist = hist.dropna(subset=["Close"])
                hist = hist.reset_index()
                hist.columns = [str(c).capitalize() for c in hist.columns]
                if len(hist['Close']) >= 14:
                    targets = TechnicalAnalysis.calculate_targets(hist, float(hist['Close'].iloc[-1]))
        except Exception:
            targets = {}
            info = {}

        target_1 = targets.get('target_1', None)
        target_2 = targets.get('target_2', None)
        target_3 = targets.get('target_3', None)

        # Determine recommendation rules:
        if target_1 is None or np.isnan(cmp_val) or cmp_val == 0:
            # fallback using P&L and heuristic
            rec = get_recommendation(pct_pnl, 0.05, cmp_val <= 0 if not pd.isna(cmp_val) else True)
            distance_text = "Target not available"
            tgt_display = "—"
        else:
            tgt_display = f"₹{float(target_1):.2f}"
            # comparison rules:
            if cmp_val <= target_1:
                rec = "BUY"
                distance_pct = (target_1 - cmp_val) / cmp_val * 100 if cmp_val else np.nan
                distance_text = f"CMP is lower than recommended by {distance_pct:.2f}%"
            elif cmp_val > target_1 and (target_2 is not None and cmp_val <= target_2):
                rec = "HOLD"
                distance_pct = (cmp_val - target_1) / target_1 * 100 if target_1 else np.nan
                distance_text = f"CMP is {distance_pct:.2f}% above the primary target (consider HOLD)"
            else:
                rec = "SELL"
                ref = target_2 if (target_2 is not None) else target_1
                distance_pct = (cmp_val - ref) / ref * 100 if ref else np.nan
                distance_text = f"CMP is higher by {distance_pct:.2f}% — CMP is still higher than recommended price"

        # Get upcoming dividend info if available from info dict
        upcoming_div_date = None
        upcoming_div_amount = None
        try:
            # yfinance 'dividendDate' is epoch int if available
            div_epoch = info.get("dividendDate") or info.get("lastDividendDate")
            if div_epoch:
                # some providers return epoch in seconds
                try:
                    dv = datetime.fromtimestamp(int(div_epoch)).date()
                except Exception:
                    # maybe it's already a date-like string
                    dv = None
                upcoming_div_date = dv

            # amount could be lastDividendValue or dividendRate or trailing annual dividend
            div_amt = info.get("lastDividendValue") or info.get("dividendRate") or info.get("trailingAnnualDividendRate")
            if div_amt:
                upcoming_div_amount = float(div_amt)
        except Exception:
            upcoming_div_date = None
            upcoming_div_amount = None

        # If we have only dividend yield, estimate amount = yield * CMP
        if upcoming_div_amount is None and info:
            try:
                dy = info.get("dividendYield")
                if dy and not np.isnan(dy) and not pd.isna(cmp_val):
                    upcoming_div_amount = float(dy) * float(cmp_val)
            except Exception:
                pass

        # Multiply per-share dividend by quantity to get expected payout
        total_upcoming_div_payout = None
        if upcoming_div_amount is not None and qty:
            try:
                total_upcoming_div_payout = upcoming_div_amount * qty
            except Exception:
                total_upcoming_div_payout = None

        # Determine suggested timeframe label using analyzer: run analyze_stock for each period and pick best score
        best_period = None
        best_score = -1
        best_summary = None
        try:
            for p in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
                res = analyze_stock(stock_name, p)
                if res and res.get("score", 0) > best_score:
                    best_score = res.get("score", 0)
                    best_period = p
                    best_summary = res
        except Exception:
            best_period = None
            best_summary = None

        timeframe_label = best_period if best_period else "N/A"

        # Render card for this stock
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown(f"<h3>{stock_name} • {rec} • {timeframe_label}</h3>", unsafe_allow_html=True)
        st.markdown(f"<div class='value'>💰 CMP: ₹{cmp_val:.2f}  | 🎯 Primary Target: {tgt_display}</div>", unsafe_allow_html=True)

        details = "<div class='chip-row'>"
        details += f"<span class='chip'>Qty: {int(qty) if qty==int(qty) else qty}</span>"
        details += f"<span class='chip'>P&L: ₹{pnl:,.2f} ({pct_pnl:.2f}%)</span>"
        if target_2:
            details += f"<span class='chip'>Target2: ₹{float(target_2):.2f}</span>"
        if target_3:
            details += f"<span class='chip'>Target3: ₹{float(target_3):.2f}</span>"
        details += "</div>"
        st.markdown(details, unsafe_allow_html=True)

        # Dividend display
        div_lines = ""
        if upcoming_div_date and upcoming_div_amount:
            div_lines = f"📅 Upcoming ex/div date: {upcoming_div_date.isoformat()} • Per-share: ₹{upcoming_div_amount:.2f}"
            if total_upcoming_div_payout is not None:
                div_lines += f" • Expected payout: ₹{total_upcoming_div_payout:,.2f} (for {int(qty)} shares)"
        elif upcoming_div_amount and not upcoming_div_date:
            div_lines = f"💸 Last/Est. dividend per share: ₹{upcoming_div_amount:.2f}"
            if total_upcoming_div_payout is not None:
                div_lines += f" • Expected payout: ₹{total_upcoming_div_payout:,.2f}"
        else:
            div_lines = "📭 No upcoming dividend info found."

        st.markdown(f"<div class='sub'>🔎 {distance_text}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='sub'>{div_lines}</div>", unsafe_allow_html=True)

        # If analyzer summary exists, show short reason snippet
        if best_summary:
            reason = best_summary.get("reasons", "")
            score = best_summary.get("score", 0)
            strength = best_summary.get("signal_strength", "")
            st.markdown(f"<div class='sub'>🧠 {timeframe_label} scanner: {strength} • Score: {score} • {reason}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- END helper for Dhan stock analysis ----------

def build_telegram_reco_message(now: datetime) -> str:
    lines = []
    lines.append("📊 Auto Stock Ideas (NIFTY 200)")
    lines.append(f"🕒 {now.strftime('%d-%m-%Y %H:%M')} IST")
    lines.append("")

    recs_state = st.session_state.get("recommendations", {})
    for period in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        recs = recs_state.get(period, [])
        if not recs:
            continue
        tmp = sorted(recs, key=lambda x: x.get("score", 0), reverse=True)[:3]
        lines.append(f"• {period}:")
        for r in tmp:
            tck = r.get("ticker", "")
            price = r.get("price", 0.0)
            strength = r.get("signal_strength", "")
            tgt = r.get("target_1", None)
            if tgt is not None:
                try:
                    tgt_fmt = f"{float(tgt):.2f}"
                except Exception:
                    tgt_fmt = str(tgt)
                lines.append(f"   - {tck}: ₹{price:.2f} → 🎯 ₹{tgt_fmt} ({strength})")
            else:
                lines.append(f"   - {tck}: ₹{price:.2f} ({strength})")
        lines.append("")
    if len(lines) <= 3:
        lines.append("No strong signals available right now.")
    lines.append("#AutoScan #NSE #Nifty200")
    return "\n".join(lines)

def send_scheduled_recommendations(now: datetime):
    if not st.session_state.get("notify_enabled", False):
        return
    recs = st.session_state.get("recommendations", {})
    if not any(recs.get(k) for k in ['BTST', 'Intraday', 'Weekly', 'Monthly']):
        run_analysis()
    msg = build_telegram_reco_message(now)
    send_telegram_message(msg)
    st.caption("📤 Telegram recommendations sent.")
    st.session_state['last_pnl_notify'] = now

def handle_scheduled_notifications(now: datetime):
    if not st.session_state.get("notify_enabled", False):
        return

    today_str = now.strftime("%Y-%m-%d")
    last_map = st.session_state.get("last_reco_notify", {}) or {}

    for slot in NOTIFY_SLOTS:
        hr, mn = map(int, slot.split(":"))
        scheduled_dt = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
        window_start = scheduled_dt
        window_end = scheduled_dt + timedelta(minutes=5)
        already_sent_today = last_map.get(slot) == today_str

        if (now >= window_start) and (now <= window_end) and not already_sent_today:
            send_scheduled_recommendations(now)
            last_map[slot] = today_str
            st.session_state['last_reco_notify'] = last_map

def auto_scan_if_due():
    now = datetime.now(IST)
    last = st.session_state.get('last_auto_scan')
    if market_hours_window(now):
        should_run = False
        if last is None:
            should_run = True
        else:
            try:
                if (now - last).total_seconds() >= 20 * 60:
                    should_run = True
            except Exception:
                should_run = True
        if should_run:
            run_analysis()
            st.caption(f"🕒 Auto-scan executed at {now.strftime('%H:%M:%S')} IST")
    handle_scheduled_notifications(now)

def get_top_stocks(limit: int = 10):
    all_recs = []
    for period in ['BTST', 'Intraday', 'Weekly', 'Monthly']:
        for r in st.session_state['recommendations'].get(period, []):
            rec = dict(r)
            rec['period'] = period
            all_recs.append(rec)
    if not all_recs:
        return []
    df_all = pd.DataFrame(all_recs).sort_values("score", ascending=False)
    seen = set()
    unique_rows = []
    for _, row in df_all.iterrows():
        t = row.get('ticker')
        if t not in seen:
            seen.add(t)
            unique_rows.append(row)
        if len(unique_rows) >= limit:
            break
    if not unique_rows:
        return []
    return pd.DataFrame(unique_rows).to_dict(orient="records")

def load_groww_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".csv"):
            df = pd.read_csv(uploaded_file, sep=None, engine="python")
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Only CSV, XLS, or XLSX files are supported.")
            return pd.DataFrame()
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return pd.DataFrame()

def map_groww_columns(df: pd.DataFrame):
    norm_cols = {c.lower().strip(): c for c in df.columns}
    required_map = {
        "stock name": "stock name",
        "isin": "isin",
        "quantity": "quantity",
        "average buy price per share": "average buy price per share",
        "total investment": "total investment",
        "total cmp": "total cmp",
        "total p&l": "total p&l",
    }
    out, missing = {}, []
    for logical_key, norm_header in required_map.items():
        if norm_header in norm_cols:
            out[logical_key] = norm_cols[norm_header]
        else:
            missing.append(logical_key)
    if missing:
        msg = (
            "Columns must match this Groww template exactly: "
            "'Stock Name, ISIN, Quantity, Average buy price per share, "
            "Total Investment, Total CMP, TOTAL P&L'. "
            "Missing or mismatched: " + ", ".join(missing)
        )
        return None, msg
    return out, None

def fetch_dividend_and_cagr(stock_name: str, isin: str, cmp_value: float):
    sym = stock_name.split()[0].upper().strip() if stock_name else ""
    yf_ticker = NIFTY_YF_MAP.get(sym, None)
    if not yf_ticker and sym:
        yf_ticker = f"{sym}.NS"
    div_yield, div_rupees, cagr = 0.0, 0.0, 0.05
    if not yf_ticker:
        return div_yield, div_rupees, cagr
    try:
        t = yf.Ticker(yf_ticker)
        info = t.info or {}
        raw_yield = info.get("dividendYield")
        if raw_yield is not None:
            div_yield = float(raw_yield)
        if cmp_value and div_yield:
            div_rupees = div_yield * cmp_value
        hist = t.history(period="10y")
        if hist is not None and not hist.empty:
            hist = hist.dropna(subset=["Close"])
            first_price = float(hist["Close"].iloc[0])
            last_price = float(hist["Close"].iloc[-1])
            years = max((hist.index[-1] - hist.index[0]).days / 365.0, 1.0)
            if first_price > 0 and years > 0:
                cagr = (last_price / first_price) ** (1.0 / years) - 1.0
    except Exception:
        pass
    return float(div_yield), float(div_rupees), float(cagr)

def classify_strength(pct_pnl: float, cagr: float, price_zero: bool) -> str:
    if price_zero:
        return "Super Strong"
    if cagr >= 0.15 and pct_pnl >= 20:
        return "Super Strong"
    if cagr >= 0.10 and pct_pnl >= 0:
        return "Strong"
    if cagr >= 0.05 or pct_pnl > -10:
        return "Medium"
    if cagr >= 0.0 or pct_pnl > -30:
        return "Weak"
    return "Super Weak"

def get_recommendation(pct_pnl: float, cagr: float, price_zero: bool) -> str:
    strength = classify_strength(pct_pnl, cagr, price_zero)
    if strength in ["Super Strong", "Strong"]:
        return "BUY"
    elif strength == "Medium":
        return "HOLD"
    else:
        return "SELL"

def suggest_horizon(strength: str, div_yield: float, cagr: float) -> str:
    if strength == "Super Strong":
        if div_yield >= 0.015 or cagr >= 0.18:
            return "Hold 20+ years (core compounding)"
        return "Hold 15–20 years"
    if strength == "Strong":
        if cagr >= 0.12:
            return "Hold 10–15 years"
        return "Hold 7–10 years"
    if strength == "Medium":
        if cagr >= 0.08:
            return "Hold 5–7 years"
        return "Review in 3–5 years"
    if strength == "Weak":
        if cagr > 0:
            return "Tactical hold; reassess within 1–2 years"
        return "Exit gradually over 1–2 years"
    return "Exit within 6–12 months; rotate to better compounders"

def main():
    st.markdown(
        '<div class="main-header"><h1>🤖 AI Stock Analysis Bot</h1>'
        '<p>Multi-timeframe scanner • 📈 NIFTY 200 • 🤝 Dhan • 📊 Groww</p>'
        '<div class="status-badge">Live • IST</div></div>',
        unsafe_allow_html=True,
    )
    sidebar_nav()
    auto_scan_if_due()
    c1, c2 = st.columns([3, 1.2])
    with c1:
        if st.button("🚀 Run Full Scan"):
            run_analysis()
    with c2:
        if st.button("🔄 Refresh View", key="refresh_btn"):
            st.rerun()
    if st.session_state['last_analysis_time']:
        st.caption(f"🕒 Last Full Scan: {st.session_state['last_analysis_time'].strftime('%d-%m-%Y %I:%M %p')}")
    st.info("On mobile you can view Top 20 stocks. For other views (BTST, Intraday, Weekly, Monthly, Groww, Dhan, Configuration), please open this dashboard on a laptop or desktop.", icon="📱")
    st.markdown("---")
    page = st.session_state['current_page']

    # ----- Main pages -----
    if page == "🔥 Top Stocks":
        st.subheader("🔥 Top Stocks (up to 20)")
        top_recs = get_top_stocks(limit=20)
        render_reco_cards(top_recs, "Top")

    elif page == "🌙 BTST":
        st.subheader("🌙 BTST Opportunities")
        recs = st.session_state['recommendations'].get('BTST', [])
        for r in recs:
            r.setdefault("period", "BTST")
        render_reco_cards(recs, "BTST")

    elif page == "⚡ Intraday":
        st.subheader("⚡ Intraday Signals")
        recs = st.session_state['recommendations'].get('Intraday', [])
        for r in recs:
            r.setdefault("period", "Intraday")
        render_reco_cards(recs, "Intraday")

    elif page == "📆 Weekly":
        st.subheader("📆 Weekly Swing Ideas")
        recs = st.session_state['recommendations'].get('Weekly', [])
        for r in recs:
            r.setdefault("period", "Weekly")
        render_reco_cards(recs, "Weekly")

    elif page == "📅 Monthly":
        st.subheader("📅 Monthly Position Trades")
        recs = st.session_state['recommendations'].get('Monthly', [])
        for r in recs:
            r.setdefault("period", "Monthly")
        render_reco_cards(recs, "Monthly")

    elif page == "📊 Groww":
        st.subheader("📊 Groww Portfolio Analysis (CSV / Excel Upload)")
        st.markdown("Upload your Groww holdings file (CSV/XLS/XLSX) to get advanced analytics with all values in ₹ INR.")
        st.code("Stock Name\tISIN\tQuantity\tAverage buy price per share\tTotal Investment\tTotal CMP\tTOTAL P&L", language="text")
        uploaded = st.file_uploader("Upload Groww portfolio file", type=["csv", "xls", "xlsx"], key="groww_file_upload")

        if uploaded is not None:
            df_up = load_groww_file(uploaded)
            if df_up.empty:
                st.stop()
            cols, err = map_groww_columns(df_up)
            if err:
                st.error(err)
                st.dataframe(df_up.head(), use_container_width=True, hide_index=True)
                st.stop()

            st.write("🔍 Raw preview:")
            st.dataframe(df_up.head(), use_container_width=True, hide_index=True)

            df = df_up.copy()
            df["_qty"] = pd.to_numeric(df[cols["quantity"]], errors="coerce").fillna(0.0)
            df["_inv"] = pd.to_numeric(df[cols["total investment"]], errors="coerce").fillna(0.0)
            df["_cmp_total"] = pd.to_numeric(df[cols["total cmp"]], errors="coerce").fillna(0.0)
            df["_pnl"] = pd.to_numeric(df[cols["total p&l"]], errors="coerce").fillna(0.0)
            df["_cmp_per_share"] = np.where(df["_qty"] > 0, df["_cmp_total"] / df["_qty"], 0.0)

            div_yields, div_rupees_list, cagr_list = [], [], []
            strength_list, pct_pnl_list, reco_list, horizon_list = [], [], [], []

            st.info("Fetching dividend yield and CAGR for each stock; defaults used if not found.")
            prog = st.progress(0)
            for i, row in df.iterrows():
                stock_name = str(row[cols["stock name"]])
                cmp_ps = float(row["_cmp_per_share"])
                is_zero_price = cmp_ps <= 0.0

                div_y, div_r, cagr = fetch_dividend_and_cagr(stock_name, str(row[cols["isin"]]), cmp_ps)
                div_yields.append(div_y)
                div_rupees_list.append(div_r)
                cagr_list.append(cagr)

                inv_val = float(row["_inv"])
                cur_val = float(row["_cmp_total"])
                pct_pnl = ((cur_val - inv_val) / inv_val * 100.0) if inv_val > 0 else 0.0
                pct_pnl_list.append(pct_pnl)

                strength = classify_strength(pct_pnl, cagr, is_zero_price)
                strength_list.append(strength)

                reco = get_recommendation(pct_pnl, cagr, is_zero_price)
                reco_list.append(reco)

                horizon = suggest_horizon(strength, div_y, cagr)
                horizon_list.append(horizon)

                prog.progress((i + 1) / len(df))
            prog.empty()

            df["Dividend Yield"] = div_yields
            df["Dividend/Share (₹)"] = div_rupees_list
            df["CAGR (decimal)"] = cagr_list
            df["CAGR (%)"] = df["CAGR (decimal)"] * 100.0
            df["Strength"] = strength_list
            df["Pct P&L (%)"] = pct_pnl_list
            df["Recommendation"] = reco_list
            df["Suggested Horizon"] = horizon_list
            df["Yearly Dividend (₹)"] = df["Dividend/Share (₹)"] * df["_qty"]

            total_inv = float(df["_inv"].sum())
            total_cmp_val = float(df["_cmp_total"].sum())
            total_pnl = float(df["_pnl"].sum())
            total_yearly_div = float(df["Yearly Dividend (₹)"].sum())

            st.markdown("### 📈 Portfolio Snapshot (All values in ₹ INR)")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Investment", f"₹{total_inv:,.2f}")
            with c2:
                st.metric("Current Value", f"₹{total_cmp_val:,.2f}")
            with c3:
                st.metric("Total P&L", f"₹{total_pnl:,.2f}")

            snap_data = [
                {"Metric": "Total Investment", "Value": f"₹{total_inv:,.2f}"},
                {"Metric": "Current Value", "Value": f"₹{total_cmp_val:,.2f}"},
                {"Metric": "Total P&L", "Value": f"₹{total_pnl:,.2f}"},
                {"Metric": "Yearly Dividend", "Value": f"₹{total_yearly_div:,.2f}"},
            ]
            snap_df = pd.DataFrame(snap_data)
            st.markdown(snap_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

            if total_cmp_val > 0:
                df["_weight"] = df["_cmp_total"] / total_cmp_val
                portfolio_cagr = float((df["CAGR (decimal)"] * df["_weight"]).sum())
            else:
                portfolio_cagr = 0.05

            st.markdown("#### 🔮 Portfolio Value Projections")
            st.markdown("*Based on weighted CAGR and total yearly dividend. All values shown in Lakhs.*")
            years_list = [1, 5, 10, 15, 20]
            proj_data = []
            for y in years_list:
                v = project_value(total_cmp_val, portfolio_cagr, total_yearly_div, y)
                capital_gain = v - total_cmp_val - (total_yearly_div * y)
                proj_data.append({
                    "Years": y,
                    "Current Value": fmt_lakhs(total_cmp_val),
                    "Total Dividends": fmt_lakhs(total_yearly_div * y),
                    "Capital Gain": fmt_lakhs(capital_gain),
                    "Projected Value": fmt_lakhs(v)
                })
            proj_df = pd.DataFrame(proj_data)
            st.markdown(proj_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

            # 🤖 AI Analysis – strength buckets
            st.markdown("### 🤖 AI Analysis – Strength Buckets for 20-Year Investing")
            st.write(
                "Each stock is classified into **Super Strong, Strong, Medium, Weak, Super Weak** "
                "based on long-term **CAGR**, **dividend yield** and your **P&L**. "
                "Use this to decide which stocks deserve a 20-year seat in your portfolio."
            )

            base_cols = [
                cols["stock name"],
                "_inv",
                "_cmp_total",
                "_pnl",
                "Dividend Yield",
                "CAGR (%)",
                "Recommendation",
                "Suggested Horizon",
            ]
            df_reco = df[base_cols + ["Strength"]].rename(columns={
                cols["stock name"]: "Stock Name",
                "_inv": "Total Investment (₹)",
                "_cmp_total": "Total CMP (₹)",
                "_pnl": "Total P&L (₹)",
                "Dividend Yield": "Dividend Yield (ratio)",
                "CAGR (%)": "CAGR (%)",
            })

            bucket_info = [
                ("Super Strong", "🌟 Core compounders for 15–20+ years"),
                ("Strong", "✅ Quality holdings for 7–15 years"),
                ("Medium", "🟡 Hold with review in 3–7 years"),
                ("Weak", "⚠️ Risky / average – reassess soon"),
                ("Super Weak", "❌ Exit candidates – capital can be better used"),
            ]

            for bucket, desc in bucket_info:
                sub = df_reco[df_reco["Strength"] == bucket]
                if sub.empty:
                    continue
                st.markdown(f"#### {bucket} — {desc}")
                st.dataframe(sub.drop(columns=["Strength"]), use_container_width=True, hide_index=True)

        else:
            st.info("Choose your Groww CSV/XLS/XLSX file to see advanced insights here.")

    elif page == "🤝 Dhan":
        st.subheader("🤝 Dhan Portfolio")
        dhan_store = localS.getItem("dhan_config") or {}
        if dhan_store:
            st.session_state['dhan_client_id'] = dhan_store.get("client_id", st.session_state['dhan_client_id'])

        dhan_enable = st.checkbox("Enable Dhan", value=st.session_state.get('dhan_enabled', False))
        st.session_state['dhan_enabled'] = dhan_enable

        if dhan_enable:
            # 🔳 Dhan configuration in black table (editable)
            current_cid = st.session_state.get('dhan_client_id', '')
            current_tok = st.session_state.get('dhan_access_token', '')

            cfg_df_edit = pd.DataFrame({
                "Field": ["Client ID", "Access Token"],
                "Value": [current_cid, current_tok],
            })
            st.markdown("#### 🧩 Dhan Configuration (edit inside table)")
            edited_cfg = st.data_editor(
                cfg_df_edit,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key="dhan_cfg_editor",
            )

            dcid = str(edited_cfg.loc[0, "Value"]) if "Value" in edited_cfg.columns else current_cid
            dtoken = str(edited_cfg.loc[1, "Value"]) if len(edited_cfg) > 1 else current_tok

            st.session_state['dhan_client_id'] = dcid
            st.session_state['dhan_access_token'] = dtoken

            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔑 Connect Dhan", use_container_width=True, key="btn_connect_dhan_main"):
                    dhan_login(dcid, dtoken)
                    localS.setItem("dhan_config", {"client_id": dcid})
            with c2:
                if st.button("🚪 Logout Dhan", use_container_width=True, key="btn_logout_dhan_main"):
                    dhan_logout()

            st.caption(st.session_state['dhan_login_msg'])

            # Summary dark table (read-only view)
            masked_token = "●" * 10 if dtoken else "Not set"
            cfg_rows = [
                {"Field": "Client ID", "Value": dcid or "Not set"},
                {"Field": "Access Token", "Value": masked_token},
                {"Field": "Status", "Value": st.session_state['dhan_login_msg']},
            ]
            cfg_df = pd.DataFrame(cfg_rows)
            st.markdown(cfg_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

            df_port, total_pnl = format_dhan_portfolio_table()
            if df_port is None or df_port.empty:
                st.info("No Dhan holdings/positions fetched yet.")
            else:
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.dataframe(df_port, use_container_width=True, hide_index=True)
                with c2:
                    st.markdown(
                        "<div class='metric-card'><h3>Total P&L</h3>"
                        "<div class='value'>₹{:,.2f}</div></div>".format(total_pnl),
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Enable Dhan above to view and refresh your portfolio.")

    elif page == "🧾 Dhan Stocks Analysis":
        # NEW tab: analyze dhan stocks and provide recommendations
        analyze_dhan_portfolio_recommendations()

    elif page == "📈 MF Analysis":
        st.subheader("📈 Mutual Fund Analysis")
        st.markdown(
            '[🔗 Open Mutual Fund Analysis App](https://airobotsmf.streamlit.app/)',
            unsafe_allow_html=False,
        )
        st.info("This opens the separate Mutual Fund AI Analysis dashboard in a new tab.")

    elif page == "⚙️ Configuration":
        st.markdown("### ⚙️ App Configuration")

        # Only Telegram configuration, in black table-editor style
        with st.expander("📨 Telegram P&L Notifications", expanded=False):
            tg_store = localS.getItem("telegram_config") or {}
            if tg_store:
                st.session_state['telegram_bot_token'] = tg_store.get("bot_token", st.session_state['telegram_bot_token'])
                st.session_state['telegram_chat_id'] = tg_store.get("chat_id", st.session_state['telegram_chat_id'])

            notify_toggle = st.checkbox(
                "Enable P&L notifications (30 min) & auto recommendations",
                value=st.session_state['notify_enabled'],
                key="cfg_notify_toggle",
            )
            st.session_state['notify_enabled'] = notify_toggle

            cur_tok = st.session_state['telegram_bot_token']
            cur_chat = st.session_state['telegram_chat_id']

            tg_df_edit = pd.DataFrame({
                "Field": ["Bot Token", "Chat ID"],
                "Value": [cur_tok, cur_chat],
            })

            st.markdown("#### 🧩 Telegram Configuration (edit inside table)")
            edited_tg = st.data_editor(
                tg_df_edit,
                hide_index=True,
                num_rows="fixed",
                use_container_width=True,
                key="tg_cfg_editor",
            )

            tg_token = str(edited_tg.loc[0, "Value"]) if "Value" in edited_tg.columns else cur_tok
            tg_chat = str(edited_tg.loc[1, "Value"]) if len(edited_tg) > 1 else cur_chat

            st.session_state['telegram_bot_token'] = tg_token
            st.session_state['telegram_chat_id'] = tg_chat

            # Summary dark table for Telegram config (no MF link here now)
            tg_rows = [
                {"Field": "Bot Token", "Value": tg_token or "Not set"},
                {"Field": "Chat ID", "Value": tg_chat or "Not set"},
                {"Field": "Notifications", "Value": "Enabled" if notify_toggle else "Disabled"},
            ]
            tg_df = pd.DataFrame(tg_rows)
            st.markdown(tg_df.to_html(classes="dark-table", index=False, escape=False), unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save settings", use_container_width=True, key="btn_save_settings"):
                    save_config_from_state()
                    localS.setItem("telegram_config", {"bot_token": tg_token, "chat_id": tg_chat})
                    st.success("Saved to config.json + browser storage")
            with c2:
                if st.button("📤 Send P&L Now", use_container_width=True, key="btn_send_pnl"):
                    text = "P&L summary feature hooked to Dhan portfolio."
                    tg_resp = send_telegram_message(text) if tg_token and tg_chat else {"info": "Telegram not configured"}
                    st.success("Triggered P&L send. Check Telegram.")
                    st.json({"telegram": tg_resp})

if __name__ == "__main__":
    main()
