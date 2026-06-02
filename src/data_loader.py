import yfinance as yf
import pandas as pd
import os

# Top NSE stocks (Yahoo Finance uses .NS suffix for NSE)
TICKERS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "ITC.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS"
]

def download_data(tickers=TICKERS, period="5y", output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    all_frames = []

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)

        if df.empty:
            print(f"  Skipping {ticker} — no data returned")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        df["Ticker"] = ticker

        df.to_csv(f"{output_dir}/{ticker.replace('.', '_')}.csv")
        all_frames.append(df)
        print(f"  {ticker}: {len(df)} rows saved")

    combined = pd.concat(all_frames)
    combined.to_csv(f"{output_dir}/all_stocks.csv")
    print(f"\nDone! Combined dataset: {len(combined)} rows")
    return combined

if __name__ == "__main__":
    download_data()