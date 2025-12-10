# 🤖 AI Stock Analysis Bot

Streamlit app for Nifty 200 technical scans, Shoonya & Dhan portfolio P&L, and deep CSV portfolio analysis with dividends and multi-horizon CAGR.

## Features

- Nifty 200 universe loaded from `data/nifty200_yahoo.csv` (NSE → Yahoo tickers).
- Multi-strategy technical scanner (RSI, MACD, Stoch, ADX, Bollinger, Volume, EMA, OBV).
- Shoonya positions P&L with optional order placement.
- Dhan holdings/positions P&L.
- CSV Portfolio Analyzer tab:
  - Upload broker CSV with:
    - `Stock Name`, `ISIN`, `Quantity`, `Average buy price per share`, `Total Investment`, `Total CMP`, `TOTAL P&L`
  - Auto-ticker normalization, yfinance price/dividend fetch, 1/3/5/10/15/20Y CAGR, and recommendations.

## Setup

# ai-stock-analysis-bot-
