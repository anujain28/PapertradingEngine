# app.py
# Updated AI Stock Analysis Bot + Paper Trading
# - Fixes SyntaxError in background loop
# - Scans NIFTY200 universe every 15 minutes (configurable)
# - Excludes tickers from Dhan & Shoonya portfolios
# - Re-checks charts/algos at time-of-buy and buys only if signal strength >= threshold
# - All paper-trading features (logs, fees, taxes, trailing, never sell losers) preserved

import os
import sys
import json
import time
import math
import threading
import traceback
from datetime import datetime, date, time as dtime, timedelta
from pathlib import Path
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import streamlit as st
import yfinance as yf
import ta
import pytz
import requests

# -------------------------
# CONFIGURATION / DEFAULTS
# -------------------------
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

NIFTY200_CSV = DATA_DIR / "nifty200_yahoo.csv"
DHAN_POS_FILE = DATA_DIR / "dhan_positions.json"
SHOONYA_POS_FILE = DATA_DIR / "shoonya_positions.json"

PAPER_PNL_FILE = DATA_DIR / "paper_pnl_log.csv"
BOOKED_TRADES_FILE = DATA_DIR / "paper_booked_trades.csv"
PAPER_POS_FILE = DATA_DIR / "paper_positions.json"
PAPER_CONFIG_FILE = DATA_DIR / "paper_config.json"

# Defaults - change as needed in UI
DEFAULT_BASE_CAPITAL = 100_000.0           # ₹1 Lakh
DEFAULT_MAX_USAGE_PCT = 0.40               # 40%
DEFAULT_MAX_POSITIONS = 5
DEFAULT_TARGET_PCT = 0.04                  # 4% target
DEFAULT_AUTO_START = False
DEFAULT_AUTO_START_TIME = dtime(hour=9, minute=30)   # 09:30 IST
DEFAULT_EOD_CAPTURE_TIME = dtime(hour=15, minute=30) # 15:30 IST
DEFAULT_TRAILING_BASE = 0.05               # default trailing pct (adaptive)
DEFAULT_BROKERAGE_PCT = 0.0003             # 0.03%
DEFAULT_GST_PCT = 0.18
DEFAULT_STT_PCT = 0.001                     # 0.1%
DEFAULT_EXCHANGE_PCT = 0.00003              # 0.003%
DEFAULT_STAMP_PCT = 0.00015                 # 0.015%
DEFAULT_PLATFORM_FEE = 10.0
DEFAULT_TAX_PCT = 0.15                      # 15% STCG

# New scan params
DEFAULT_SCAN_INTERVAL_MIN = 15
DEFAULT_BUY_SCORE_THRESHOLD = 65.0

IST = pytz.timezone("Asia/Kolkata")

# Keep-alive / self ping
KEEP_ALIVE = True
SELF_URL = "https://airobots.streamlit.app/"   # user provided; can be changed in UI
SELF_PING_INTERVAL_SEC = 4 * 60  # 4 minutes

# -------------------------
# Utilities
# -------------------------
def now_ist():
    return datetime.now(IST)

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default

def save_json(path: Path, obj):
    try:
        with path.open("w") as f:
            json.dump(obj, f, default=str, indent=2)
    except Exception:
        pass

def load_json(path: Path):
    try:
        if path.exists():
            with path.open("r") as f:
                return json.load(f)
    except Exception:
        pass
    return None

def append_csv(path: Path, row: Dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, mode="w", header=True, index=False)

# -------------------------
# Load stock universe
# -------------------------
def load_nifty200_universe():
    symbols = []
    mapping = {}
    if NIFTY200_CSV.exists():
        try:
            df = pd.read_csv(NIFTY200_CSV)
            if "SYMBOL" in df.columns:
                df["SYMBOL"] = df["SYMBOL"].astype(str).str.strip().str.upper()
                symbols = df["SYMBOL"].dropna().unique().tolist()
            if "YF_TICKER" in df.columns and "SYMBOL" in df.columns:
                mapping = dict(zip(df["SYMBOL"].astype(str).str.strip().str.upper(), df["YF_TICKER"].astype(str).str.strip()))
        except Exception:
            symbols = []
            mapping = {}
    if not symbols:
        # fallback sample (small set)
        symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "AXISBANK", "BAJFINANCE", "LT"]
    return symbols, mapping

STOCK_UNIVERSE, STOCK_YF_MAP = load_nifty200_universe()

def nse_yf_symbol(sym: str) -> str:
    if not sym:
        return ""
    s = str(sym).strip().upper()
    if s in STOCK_YF_MAP:
        return STOCK_YF_MAP[s]
    return s if s.endswith(".NS") else f"{s}.NS"

# -------------------------
# Fetch data & indicators
# -------------------------
def fetch_yf_df(symbol: str, period: str = "60d", interval: str = "15m"):
    try:
        t = yf.Ticker(nse_yf_symbol(symbol))
        df = t.history(period=period, interval=interval, auto_adjust=True)
        return df
    except Exception:
        return pd.DataFrame()

def compute_indicators(df: pd.DataFrame):
    out = {}
    if df is None or df.empty:
        return out
    c = df["Close"]
    h = df["High"] if "High" in df else c
    l = df["Low"] if "Low" in df else c
    try:
        out['rsi'] = ta.momentum.rsi(c, window=14).iloc[-1]
    except Exception:
        out['rsi'] = np.nan
    try:
        macd = ta.trend.MACD(c)
        out['macd'] = safe_float(macd.macd().iloc[-1])
        out['macd_sig'] = safe_float(macd.macd_signal().iloc[-1])
        out['macd_diff'] = safe_float(macd.macd_diff().iloc[-1])
    except Exception:
        out['macd'] = out['macd_sig'] = out['macd_diff'] = np.nan
    try:
        atr = ta.volatility.average_true_range(h, l, c, window=14).iloc[-1]
        out['atr'] = safe_float(atr)
        out['atr_pct'] = safe_float(atr) / float(c.iloc[-1]) * 100.0 if c.iloc[-1] else np.nan
    except Exception:
        out['atr'] = out['atr_pct'] = np.nan
    try:
        bb = ta.volatility.BollingerBands(c, window=20, window_dev=2)
        out['bb_width'] = safe_float(bb.bollinger_hband().iloc[-1] - bb.bollinger_lband().iloc[-1]) / float(c.iloc[-1]) * 100.0
    except Exception:
        out['bb_width'] = np.nan
    try:
        vol = df["Volume"] if "Volume" in df else pd.Series(dtype=float)
        out['vol'] = float(vol.iloc[-1]) if not vol.empty else np.nan
        out['vol_avg20'] = float(vol.rolling(20).mean().iloc[-1]) if len(vol) >= 20 else np.nan
    except Exception:
        out['vol'] = out['vol_avg20'] = np.nan
    try:
        out['ema9'] = float(ta.trend.ema_indicator(c, window=9).iloc[-1])
        out['ema21'] = float(ta.trend.ema_indicator(c, window=21).iloc[-1])
    except Exception:
        out['ema9'] = out['ema21'] = np.nan
    # on-balance volume
    try:
        obv = ta.volume.OnBalanceVolumeIndicator(c, df["Volume"]).on_balance_volume()
        out['obv'] = safe_float(obv.iloc[-1])
        out['obv_sma10'] = safe_float(obv.rolling(10).mean().iloc[-1])
    except Exception:
        out['obv'] = out['obv_sma10'] = np.nan
    return out

# -------------------------
# Scoring function (used for buy decision)
# -------------------------
def compute_signal_score_from_indicators(ind: Dict, df: pd.DataFrame, entry_price: Optional[float]=None):
    """
    Returns a 0-100 score representing signal strength.
    Heuristics:
      - RSI low (<40) adds score
      - MACD bullish (macd > macd_sig) adds score
      - EMA9 > EMA21 adds score
      - Volume spike adds score
      - OBV accumulation adds score
      - Bollinger squeeze breakout adds some score
    """
    score = 0.0
    try:
        rsi = ind.get("rsi", np.nan)
        if not np.isnan(rsi):
            if rsi < 30:
                score += 22
            elif rsi < 40:
                score += 14
            elif rsi < 50:
                score += 6

        macd = ind.get("macd", np.nan)
        macd_sig = ind.get("macd_sig", np.nan)
        macd_diff = ind.get("macd_diff", np.nan)
        if not np.isnan(macd) and not np.isnan(macd_sig):
            if macd > macd_sig and macd_diff > 0:
                score += 20

        ema9 = ind.get("ema9", np.nan)
        ema21 = ind.get("ema21", np.nan)
        if not np.isnan(ema9) and not np.isnan(ema21):
            if ema9 > ema21:
                score += 12

        vol = ind.get("vol", np.nan)
        vol20 = ind.get("vol_avg20", np.nan)
        if not np.isnan(vol) and not np.isnan(vol20):
            if vol > vol20 * 1.5:
                score += 12

        obv = ind.get("obv", np.nan)
        obv_sma = ind.get("obv_sma10", np.nan)
        if not np.isnan(obv) and not np.isnan(obv_sma):
            if obv > obv_sma:
                score += 8

        bbw = ind.get("bb_width", np.nan)
        if not np.isnan(bbw):
            if bbw < 5:  # small width then breakout is more meaningful
                score += 6

        # price proximity to entry (if provided) - reward if fresh breakout above entry or support
        if entry_price:
            latest_price = float(df["Close"].iloc[-1])
            dist = (latest_price - entry_price) / entry_price * 100.0
            if dist > 0:
                score += min(10, dist) * 0.5

    except Exception:
        score = 0.0
    return min(100.0, score)

# -------------------------
# Paper trading engine (same as before, cleaned)
# -------------------------
class PaperTrader:
    def __init__(self):
        self.positions = load_json(PAPER_POS_FILE) or {}
        self.booked_file = BOOKED_TRADES_FILE
        self.pnl_file = PAPER_PNL_FILE
        self.lock = threading.Lock()

    def persist_positions(self):
        save_json(PAPER_POS_FILE, self.positions)

    def snapshot_positions_df(self) -> pd.DataFrame:
        if not self.positions:
            return pd.DataFrame()
        rows = []
        for t, v in self.positions.items():
            rows.append({
                "Ticker": t,
                "Qty": v.get("qty"),
                "EntryPrice": v.get("entry_price"),
                "Invested": v.get("invested"),
                "Peak": v.get("peak", v.get("entry_price")),
                "EntryTime": v.get("entry_time"),
                "Source": v.get("source", "auto")
            })
        return pd.DataFrame(rows)

    def buy(self, ticker: str, alloc: float, price: float, qty: int, source: str = "auto"):
        with self.lock:
            if qty <= 0:
                return None
            invested = price * qty
            self.positions[ticker] = {
                "entry_price": float(price),
                "qty": int(qty),
                "invested": float(invested),
                "peak": float(price),
                "entry_time": datetime.utcnow().isoformat(),
                "source": source,
            }
            self.persist_positions()
            return self.positions[ticker]

    def update_peak_and_mark(self, ticker: str, price: float):
        with self.lock:
            pos = self.positions.get(ticker)
            if not pos:
                return
            if price > pos.get("peak", pos["entry_price"]):
                pos["peak"] = float(price)
            self.positions[ticker] = pos
            self.persist_positions()

    def eligible_to_sell(self, ticker: str, price: float, trailing_pct: float):
        pos = self.positions.get(ticker)
        if not pos:
            return False, "no_position"
        entry = pos["entry_price"]
        if price <= entry:
            return False, "in_loss_no_sell"
        peak = pos.get("peak", entry)
        if price <= peak * (1 - trailing_pct):
            return True, "trailing_trigger"
        return False, "no_trigger"

    def sell(self, ticker: str, price: float, fees_cfg: Dict, reason: str = "auto-trail"):
        with self.lock:
            pos = self.positions.get(ticker)
            if not pos:
                return None
            qty = int(pos["qty"])
            invested = float(pos["invested"])
            sell_turnover = price * qty
            gross_profit = sell_turnover - invested

            brokerage = sell_turnover * fees_cfg.get("brokerage_pct", DEFAULT_BROKERAGE_PCT)
            gst = brokerage * fees_cfg.get("gst_pct", DEFAULT_GST_PCT)
            stt = sell_turnover * fees_cfg.get("stt_pct", DEFAULT_STT_PCT)
            exchange = sell_turnover * fees_cfg.get("exchange_pct", DEFAULT_EXCHANGE_PCT)
            stamp = sell_turnover * fees_cfg.get("stamp_pct", DEFAULT_STAMP_PCT)
            platform_fee = fees_cfg.get("platform_fee", DEFAULT_PLATFORM_FEE)
            fees_sum = brokerage + gst + stt + exchange + stamp + platform_fee

            tax = max(0.0, gross_profit) * fees_cfg.get("tax_pct", DEFAULT_TAX_PCT)

            net_realized = gross_profit - fees_sum - tax

            booked = {
                "timestamp": datetime.utcnow().isoformat(),
                "ticker": ticker,
                "qty": qty,
                "entry_price": pos["entry_price"],
                "sell_price": float(price),
                "invested": invested,
                "turnover": sell_turnover,
                "gross_profit": gross_profit,
                "brokerage": brokerage,
                "gst": gst,
                "stt": stt,
                "exchange_fee": exchange,
                "stamp_duty": stamp,
                "platform_fee": platform_fee,
                "fees_sum": fees_sum,
                "tax": tax,
                "net_realized": net_realized,
                "reason": reason
            }
            try:
                append_csv(Path(self.booked_file), booked)
                append_csv(Path(self.pnl_file), {
                    "date": date.today().isoformat(),
                    "timestamp": booked["timestamp"],
                    "ticker": ticker,
                    "net_realized": net_realized,
                    "gross_profit": gross_profit
                })
            except Exception:
                pass

            try:
                del self.positions[ticker]
            except Exception:
                pass
            self.persist_positions()
            return booked

    def load_open_positions(self):
        return self.positions

paper = PaperTrader()

# -------------------------
# Scheduler with scan & buy logic
# -------------------------
class PaperTradingScheduler:
    def __init__(self):
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.config = {
            "base_capital": DEFAULT_BASE_CAPITAL,
            "max_usage_pct": DEFAULT_MAX_USAGE_PCT,
            "max_positions": DEFAULT_MAX_POSITIONS,
            "target_pct": DEFAULT_TARGET_PCT,
            "trailing_base": DEFAULT_TRAILING_BASE,
            "brokerage_pct": DEFAULT_BROKERAGE_PCT,
            "gst_pct": DEFAULT_GST_PCT,
            "stt_pct": DEFAULT_STT_PCT,
            "exchange_pct": DEFAULT_EXCHANGE_PCT,
            "stamp_pct": DEFAULT_STAMP_PCT,
            "platform_fee": DEFAULT_PLATFORM_FEE,
            "tax_pct": DEFAULT_TAX_PCT,
            "auto_start": DEFAULT_AUTO_START,
            "auto_start_time": DEFAULT_AUTO_START_TIME,
            "eod_capture_time": DEFAULT_EOD_CAPTURE_TIME,
            "max_positions": DEFAULT_MAX_POSITIONS,
            "self_url": SELF_URL,
            "keep_alive": KEEP_ALIVE,
            "scan_interval_min": DEFAULT_SCAN_INTERVAL_MIN,
            "buy_score_threshold": DEFAULT_BUY_SCORE_THRESHOLD
        }
        self.load_config()

    def config_path(self):
        return PAPER_CONFIG_FILE

    def save_config(self):
        save_json(self.config_path(), self.config)

    def load_config(self):
        cfg = load_json(self.config_path())
        if cfg:
            self.config.update(cfg)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def _get_exclusion_list(self):
        exc = set()
        try:
            d = load_json(DHAN_POS_FILE)
            if isinstance(d, dict):
                exc.update([k.upper() for k in d.keys()])
            elif isinstance(d, list):
                exc.update([str(x).upper() for x in d])
        except Exception:
            pass
        try:
            s = load_json(SHOONYA_POS_FILE)
            if isinstance(s, dict):
                exc.update([k.upper() for k in s.keys()])
            elif isinstance(s, list):
                exc.update([str(x).upper() for x in s])
        except Exception:
            pass
        return exc

    def _scan_and_buy_cycle(self):
        """
        Scan universe, exclude Dhan/Shoonya and already-held positions, compute scores, pick top candidates.
        For each candidate: re-check indicators & score live; if >= buy_score_threshold -> buy (paper)
        """
        try:
            universe = STOCK_UNIVERSE.copy()
            # uppercase
            universe = [u.upper() for u in universe]
            exclusion = self._get_exclusion_list()
            # exclude current open positions
            open_pos = paper.load_open_positions() or {}
            exclusion.update([k.upper() for k in open_pos.keys()])
            # filter
            candidates = [s for s in universe if s.upper() not in exclusion]
            if not candidates:
                return

            scored = []
            for sym in candidates:
                try:
                    df = fetch_yf_df(sym, period="90d", interval="1d")
                    if df is None or df.empty or len(df) < 30:
                        continue
                    ind = compute_indicators(df)
                    score = compute_signal_score_from_indicators(ind, df)
                    scored.append((sym, score, ind, df))
                except Exception:
                    continue

            if not scored:
                return

            # pick top few by score
            scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
            picks = []
            for sym, score, ind, df in scored_sorted[: int(self.config.get("max_positions", DEFAULT_MAX_POSITIONS) * 2)]:  # pick some extras to re-validate later
                picks.append({"ticker": sym, "score": score, "ind": ind, "df": df})

            # Now re-validate each top pick with fresh intraday data and indicators before buying
            to_buy = []
            for p in picks:
                sym = p["ticker"]
                # get fresh short-term data (intraday) to check live conditions
                df_live = fetch_yf_df(sym, period="5d", interval="5m")
                if df_live is None or df_live.empty:
                    continue
                ind_live = compute_indicators(df_live)
                live_score = compute_signal_score_from_indicators(ind_live, df_live)
                threshold = float(self.config.get("buy_score_threshold", DEFAULT_BUY_SCORE_THRESHOLD))
                # require live score >= threshold and previous scan score > some baseline
                if live_score >= threshold and p["score"] >= (threshold - 5):
                    # candidate price: last close
                    price = float(df_live["Close"].iloc[-1])
                    to_buy.append({"ticker": sym, "price": price, "source": "scan", "score": live_score})

            # allocate available capital across up to max_positions
            if not to_buy:
                return

            # respect max positions and available capital
            max_positions = int(self.config.get("max_positions", DEFAULT_MAX_POSITIONS))
            to_buy = to_buy[:max_positions]

            # run buy
            self.run_selection_and_buy(to_buy)
        except Exception:
            # swallow exceptions in scheduler to keep it running
            traceback.print_exc()

    def run_selection_and_buy(self, picks: List[Dict], config_override: Dict = None):
        try:
            cfg = self.config
            base = float(cfg["base_capital"])
            max_used = base * float(cfg["max_usage_pct"])
            current_used = sum([v.get("invested", 0.0) for v in (paper.load_open_positions() or {}).values()])
            available_for_new = max(0.0, max_used - current_used)
            picks = picks[: int(cfg["max_positions"])]
            if not picks or available_for_new <= 0:
                return []

            per_pick_alloc = available_for_new / len(picks)
            bought = []
            for p in picks:
                t = p.get("ticker")
                price = float(p.get("price", p.get("cmp", 0.0)))
                if price <= 0:
                    continue
                qty = int(math.floor(per_pick_alloc / price))
                if qty <= 0:
                    continue
                invested = qty * price
                b = paper.buy(t, per_pick_alloc, price, qty, source=p.get("source","scan"))
                if b:
                    bought.append({"ticker": t, "qty": qty, "buy_price": price, "invested": invested})
            return bought
        except Exception:
            traceback.print_exc()
            return []

    def _loop(self):
        last_scan_ts = None
        scan_interval_min = int(self.config.get("scan_interval_min", DEFAULT_SCAN_INTERVAL_MIN))
        while self.running:
            try:
                now = now_ist()

                # update peaks for all open positions
                positions = paper.load_open_positions() or {}
                for tck, pos in list(positions.items()):
                    try:
                        df = fetch_yf_df(tck, period="5d", interval="5m")
                        if df is None or df.empty:
                            continue
                        last_price = float(df["Close"].iloc[-1])
                        paper.update_peak_and_mark(tck, last_price)

                        # evaluate trailing & AI-book for existing positions
                        inds = compute_indicators(df)
                        inds['peak'] = pos.get("peak", pos["entry_price"])
                        # trailing pct adaptive
                        atr_pct = inds.get("atr_pct", np.nan)
                        if not np.isnan(atr_pct):
                            if atr_pct >= 3.0:
                                trailing_pct = 0.05
                            elif atr_pct >= 2.0:
                                trailing_pct = 0.04
                            elif atr_pct >= 1.0:
                                trailing_pct = 0.03
                            else:
                                trailing_pct = 0.02
                        else:
                            trailing_pct = self.config.get("trailing_base", DEFAULT_TRAILING_BASE)

                        # trailing trigger check
                        can_sell, reason = paper.eligible_to_sell(tck, last_price, trailing_pct)
                        if can_sell:
                            fees_cfg = {
                                "brokerage_pct": self.config["brokerage_pct"],
                                "gst_pct": self.config["gst_pct"],
                                "stt_pct": self.config["stt_pct"],
                                "exchange_pct": self.config["exchange_pct"],
                                "stamp_pct": self.config["stamp_pct"],
                                "platform_fee": self.config["platform_fee"],
                                "tax_pct": self.config["tax_pct"]
                            }
                            paper.sell(tck, last_price, fees_cfg, reason=f"trailing_{int(trailing_pct*100)}")
                        else:
                            # compute book-now score
                            # time to eod
                            eod_dt = datetime.combine(now.date(), self.config.get("eod_capture_time", DEFAULT_EOD_CAPTURE_TIME))
                            eod_dt = IST.localize(eod_dt)
                            time_to_eod = max(0, int((eod_dt - now).total_seconds() // 60))
                            book_score = compute_signal_score_from_indicators(inds, df, entry_price=pos.get("entry_price"))
                            if book_score >= float(self.config.get("buy_score_threshold", DEFAULT_BUY_SCORE_THRESHOLD)):
                                # only book if profitable
                                if last_price > pos.get("entry_price"):
                                    fees_cfg = {
                                        "brokerage_pct": self.config["brokerage_pct"],
                                        "gst_pct": self.config["gst_pct"],
                                        "stt_pct": self.config["stt_pct"],
                                        "exchange_pct": self.config["exchange_pct"],
                                        "stamp_pct": self.config["stamp_pct"],
                                        "platform_fee": self.config["platform_fee"],
                                        "tax_pct": self.config["tax_pct"]
                                    }
                                    paper.sell(tck, last_price, fees_cfg, reason=f"ai_early_book_{int(book_score)}")
                    except Exception:
                        continue

                # run a full universe scan every scan_interval_min
                if last_scan_ts is None or (now - last_scan_ts).total_seconds() >= scan_interval_min * 60:
                    self._scan_and_buy_cycle()
                    last_scan_ts = now

                # EOD snapshot capture (only near EOD)
                now_time = now.time()
                eod_time = self.config.get("eod_capture_time", DEFAULT_EOD_CAPTURE_TIME)
                # capture within +/- 65 seconds of eod_time
                eod_dt = datetime.combine(now.date(), eod_time)
                eod_dt = IST.localize(eod_dt)
                if abs((now - eod_dt).total_seconds()) < 65:
                    # compute unrealised and realized today
                    unreal = 0.0
                    for tck, pos in (paper.load_open_positions() or {}).items():
                        try:
                            df = fetch_yf_df(tck, period="5d", interval="5m")
                            if df is None or df.empty:
                                continue
                            last_price = float(df["Close"].iloc[-1])
                            unreal += (last_price * pos["qty"] - pos["invested"])
                        except Exception:
                            continue
                    realized_today = 0.0
                    try:
                        if Path(self.pnl_file).exists():
                            pnl_df = pd.read_csv(self.pnl_file)
                            realized_today = float(pnl_df[pnl_df['date'] == date.today().isoformat()]['net_realized'].sum())
                    except Exception:
                        realized_today = 0.0
                    snapshot = {
                        "date": date.today().isoformat(),
                        "timestamp": datetime.utcnow().isoformat(),
                        "realized_today": realized_today,
                        "unrealised": unreal,
                        "total_capital": float(self.config["base_capital"]) + realized_today + unreal
                    }
                    append_csv(Path(self.pnl_file), {
                        "date": snapshot["date"],
                        "timestamp": snapshot["timestamp"],
                        "ticker": "SNAPSHOT",
                        "net_realized": snapshot["realized_today"]
                    })
                    # sleep to avoid multiple captures within the window
                    time.sleep(70)

                time.sleep(10)

            except Exception:
                # log and sleep a bit to avoid tight exception loop
                traceback.print_exc()
                time.sleep(5)

    def run_selection_and_buy_manual(self, picks: List[Dict]):
        return self.run_selection_and_buy(picks)

    def run_selection_and_buy(self, picks: List[Dict], config_override: Dict = None):
        # same as earlier implementation
        try:
            cfg = self.config
            base = float(cfg["base_capital"])
            max_used = base * float(cfg["max_usage_pct"])
            current_used = sum([v.get("invested", 0.0) for v in (paper.load_open_positions() or {}).values()])
            available_for_new = max(0.0, max_used - current_used)
            picks = picks[: int(cfg["max_positions"])]
            if not picks or available_for_new <= 0:
                return []

            per_pick_alloc = available_for_new / len(picks)
            bought = []
            for p in picks:
                t = p.get("ticker")
                price = float(p.get("price", p.get("cmp", 0.0)))
                if price <= 0:
                    continue
                qty = int(math.floor(per_pick_alloc / price))
                if qty <= 0:
                    continue
                invested = qty * price
                b = paper.buy(t, per_pick_alloc, price, qty, source=p.get("source","scan"))
                if b:
                    bought.append({"ticker": t, "qty": qty, "buy_price": price, "invested": invested})
            return bought
        except Exception:
            traceback.print_exc()
            return []

paper_scheduler = PaperTradingScheduler()

# -------------------------
# Self-pinger (keep alive)
# -------------------------
def self_ping_loop(url: str, interval: int, enabled: bool):
    if not enabled or not url:
        return
    while True:
        try:
            requests.get(url, timeout=10)
        except Exception:
            pass
        time.sleep(interval)

if KEEP_ALIVE and SELF_URL:
    tping = threading.Thread(target=self_ping_loop, args=(SELF_URL, SELF_PING_INTERVAL_SEC, KEEP_ALIVE), daemon=True)
    tping.start()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="🤖 AI Stock Analysis Bot + Paper Trading", page_icon="📈", layout="wide")
st.markdown("<style> .stApp { background-color: #fbfbf9; } </style>", unsafe_allow_html=True)

# session state defaults
if "paper_running" not in st.session_state:
    st.session_state["paper_running"] = False
if "paper_auto_start" not in st.session_state:
    st.session_state["paper_auto_start"] = paper_scheduler.config.get("auto_start", DEFAULT_AUTO_START)
if "base_capital" not in st.session_state:
    st.session_state["base_capital"] = paper_scheduler.config.get("base_capital", DEFAULT_BASE_CAPITAL)

# Sidebar navigation
NAV_PAGES = [
    "🔥 Top Stocks",
    "🌙 BTST",
    "⚡ Intraday",
    "📆 Weekly",
    "📅 Monthly",
    "📣 Dividends",
    "📊 Groww",
    "🤝 Dhan",
    "🧾 Dhan Stocks Analysis",
    "🧾 PNL Log",
    "🪙 Paper Trading",
    "⚙️ Configuration",
]
with st.sidebar:
    page = st.radio("Navigation", NAV_PAGES, index=NAV_PAGES.index("🔥 Top Stocks"))

# Header
st.markdown('<div style="padding:12px;border-radius:12px;background:linear-gradient(90deg,#4f46e5,#0ea5e9);color:white"><h2>AI Stock Analysis Bot + Paper Trading</h2></div>', unsafe_allow_html=True)
st.caption(f"Local time: {now_ist().strftime('%d-%m-%Y %I:%M %p')} IST")

# Pages
if page == "🔥 Top Stocks":
    st.subheader("🔥 Top Stocks (preview)")
    st.write("Run scans from Paper Trading or use the analyzer to populate recommendations.")

elif page == "📣 Dividends":
    st.subheader("📣 Upcoming Dividends demo")
    st.info("Dividends scanning available in full app; simplified here.")

elif page == "📊 Groww":
    st.subheader("📊 Groww Upload (demo)")
    st.info("Upload Groww CSV to analyze.")
    uploaded = st.file_uploader("Upload Groww portfolio file", type=["csv", "xls", "xlsx"])
    if uploaded:
        try:
            df_up = pd.read_csv(uploaded) if str(uploaded.name).lower().endswith(".csv") else pd.read_excel(uploaded)
            st.write(df_up.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")

elif page == "🤝 Dhan":
    st.subheader("🤝 Dhan demo")
    st.info("Dhan integration is supported in full app; this demo shows a placeholder.")

elif page == "🧾 Dhan Stocks Analysis":
    st.subheader("🧾 Dhan Stocks Analysis (demo)")
    st.info("This page will show Dhan holdings + AI recommendations in full app.")
    df_pos = paper.snapshot_positions_df()
    if df_pos.empty:
        st.info("No paper positions yet. Start Paper Trading to see holdings here.")
    else:
        st.dataframe(df_pos)

elif page == "🧾 PNL Log":
    st.subheader("🧾 Paper Trading PNL Log")
    if Path(BOOKED_TRADES_FILE).exists():
        df_booked = pd.read_csv(BOOKED_TRADES_FILE)
        st.markdown("### Booked Trades (realized)")
        st.dataframe(df_booked.sort_values("timestamp", ascending=False).head(200))
        st.download_button("Download Booked Trades CSV", data=df_booked.to_csv(index=False), file_name="booked_trades.csv")
    else:
        st.info("No booked trades yet.")

    if Path(PAPER_PNL_FILE).exists():
        df_pnl = pd.read_csv(PAPER_PNL_FILE)
        st.markdown("### PnL Log")
        st.dataframe(df_pnl.sort_values(["date","timestamp"], ascending=False).head(300))
        st.download_button("Download PnL Log CSV", data=df_pnl.to_csv(index=False), file_name="paper_pnl_log.csv")
    else:
        st.info("No PnL snapshots yet.")

elif page == "🪙 Paper Trading":
    st.subheader("🪙 Paper Trading Engine")
    st.markdown("Configure and start the simulator. It scans NIFTY200 every scan interval and buys only if live signal strength >= threshold.")

    # Controls
    c1, c2 = st.columns([2, 3])
    with c1:
        base_cap = st.number_input("Base capital (₹)", value=float(st.session_state.get("base_capital", DEFAULT_BASE_CAPITAL)), step=1000.0)
        max_usage = st.slider("Max capital usage (%)", min_value=10, max_value=100, value=int(DEFAULT_MAX_USAGE_PCT*100))/100.0
        max_pos = st.number_input("Max positions", min_value=1, max_value=10, value=int(DEFAULT_MAX_POSITIONS))
        target_pct = st.number_input("Target % (per trade)", value=float(DEFAULT_TARGET_PCT*100.0))/100.0
    with c2:
        trailing_base = st.number_input("Default trailing % (fallback)", value=float(DEFAULT_TRAILING_BASE*100.0))/100.0
        scan_interval = st.number_input("Scan interval (minutes)", value=int(paper_scheduler.config.get("scan_interval_min", DEFAULT_SCAN_INTERVAL_MIN)), min_value=5, max_value=60, step=5)
        buy_threshold = st.number_input("Buy score threshold (0-100)", value=float(paper_scheduler.config.get("buy_score_threshold", DEFAULT_BUY_SCORE_THRESHOLD)))
        brokerage_pct = st.number_input("Brokerage %", value=float(DEFAULT_BROKERAGE_PCT*100.0))/100.0
        stt_pct = st.number_input("STT % (sell)", value=float(DEFAULT_STT_PCT*100.0))/100.0
        tax_pct = st.number_input("Tax % on gain", value=float(DEFAULT_TAX_PCT*100.0))/100.0

    # apply to scheduler config
    paper_scheduler.config.update({
        "base_capital": float(base_cap),
        "max_usage_pct": float(max_usage),
        "max_positions": int(max_pos),
        "target_pct": float(target_pct),
        "trailing_base": float(trailing_base),
        "brokerage_pct": float(brokerage_pct),
        "gst_pct": float(DEFAULT_GST_PCT),
        "stt_pct": float(stt_pct),
        "tax_pct": float(tax_pct),
        "scan_interval_min": int(scan_interval),
        "buy_score_threshold": float(buy_threshold)
    })
    paper_scheduler.save_config()

    # Start / Stop buttons
    cola, colb = st.columns(2)
    with cola:
        if st.button("Start Paper Trading"):
            if not st.session_state["paper_running"]:
                paper_scheduler.start()
                st.session_state["paper_running"] = True
                st.success("Paper Trading started (background scheduler running).")
            else:
                st.info("Paper Trading already running.")
    with colb:
        if st.button("Stop Paper Trading"):
            if st.session_state["paper_running"]:
                paper_scheduler.stop()
                st.session_state["paper_running"] = False
                st.success("Paper Trading stopped.")
            else:
                st.info("Paper Trading not running.")

    # Manual quick selection run (uses live scan)
    if st.button("Run universe scan now (manual)"):
        st.info("Running quick scan (this may take some seconds)...")
        paper_scheduler._scan_and_buy_cycle()
        st.success("Manual scan done. Check Open Positions / Booked Trades.")

    # show open positions
    st.markdown("### Open Paper Positions")
    df_open = paper.snapshot_positions_df()
    if df_open.empty:
        st.info("No open paper positions.")
    else:
        st.dataframe(df_open)
        if st.button("Force Book All Profitable (manual)"):
            pos = paper.load_open_positions() or {}
            cnt = 0
            for tck, p in list(pos.items()):
                try:
                    df = fetch_yf_df(tck, period="5d", interval="5m")
                    if df is None or df.empty:
                        continue
                    last = float(df["Close"].iloc[-1])
                    if last > p["entry_price"]:
                        booked = paper.sell(tck, last, {
                            "brokerage_pct": paper_scheduler.config["brokerage_pct"],
                            "gst_pct": paper_scheduler.config["gst_pct"],
                            "stt_pct": paper_scheduler.config["stt_pct"],
                            "exchange_pct": paper_scheduler.config["exchange_pct"],
                            "stamp_pct": paper_scheduler.config["stamp_pct"],
                            "platform_fee": paper_scheduler.config["platform_fee"],
                            "tax_pct": paper_scheduler.config["tax_pct"],
                        }, reason="manual_force_book")
                        cnt += 1
                except Exception:
                    continue
            st.success(f"Booked {cnt} profitable positions (manual).")

elif page == "⚙️ Configuration":
    st.subheader("⚙️ Configuration & Keep-Alive")
    st.markdown("Configure app behavior. Keep the self-url set to your deployed app if you want the internal pinger to try to keep it awake.")
    self_url = st.text_input("Self URL (for keep-alive pinger)", value=paper_scheduler.config.get("self_url", SELF_URL))
    keep_alive = st.checkbox("Enable internal self-ping keep-alive (may not prevent host sleep on free tiers)", value=paper_scheduler.config.get("keep_alive", KEEP_ALIVE))
    if st.button("Save Keep-Alive"):
        paper_scheduler.config["self_url"] = self_url
        paper_scheduler.config["keep_alive"] = bool(keep_alive)
        paper_scheduler.save_config()
        st.success("Saved keep-alive config (note: external Uptime pinger is more reliable).")

st.markdown("---")
st.caption("Paper Trading is simulation-only. The engine will not place real orders. Check booked trades & PnL log from the sidebar.")

