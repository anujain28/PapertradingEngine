import os
import io
import csv
import requests
import pandas as pd

NIFTY200_SOURCE_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"  # official Nifty 200 list[web:62]
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "nifty200_yahoo.csv")

def download_nifty200_csv(url: str) -> pd.DataFrame:
    """
    Download Nifty 200 constituents CSV from NSE/Nifty Indices and return as DataFrame.
    Expected columns typically include: Company Name, Industry, Symbol, Series, ISIN, etc.[web:61][web:62]
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "text/csv,application/csv,application/octet-stream",
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    # Some NSE CSVs are not perfectly RFC‑compliant; use pandas to parse robustly
    content = resp.content.decode("utf-8", errors="ignore")
    df = pd.read_csv(io.StringIO(content))
    return df

def build_yahoo_mapping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a SYMBOL → YF_TICKER mapping.
    For most NSE cash symbols, Yahoo ticker is SYMBOL.NS, so we use that rule.[web:66]
    """
    # Try to locate the symbol column
    symbol_col = None
    for candidate in ["Symbol", "SYMBOL", "symbol"]:
        if candidate in df.columns:
            symbol_col = candidate
            break
    if symbol_col is None:
        raise ValueError(f"Could not find symbol column in Nifty 200 CSV. Columns: {list(df.columns)}")

    symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .replace("", pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    rows = []
    for sym in symbols:
        yf_ticker = f"{sym}.NS"  # default NSE → Yahoo rule
        rows.append({"SYMBOL": sym, "YF_TICKER": yf_ticker})

    out_df = pd.DataFrame(rows).sort_values("SYMBOL").reset_index(drop=True)
    return out_df

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Downloading Nifty 200 list from: {NIFTY200_SOURCE_URL}")
    src_df = download_nifty200_csv(NIFTY200_SOURCE_URL)
    print(f"Downloaded {len(src_df)} rows from source.")

    map_df = build_yahoo_mapping(src_df)
    print(f"Built mapping for {len(map_df)} symbols.")

    map_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved Yahoo mapping to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
