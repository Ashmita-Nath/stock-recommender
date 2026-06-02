import pandas as pd
import numpy as np

def add_technical_indicators(df):
    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Trend indicators ──────────────────────────────────────────
    df["SMA_20"]  = close.rolling(20).mean()
    df["SMA_50"]  = close.rolling(50).mean()
    df["EMA_20"]  = close.ewm(span=20, adjust=False).mean()
    df["MACD"]    = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    # ── Momentum indicators ───────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["ROC_10"] = close.pct_change(10) * 100
    df["MOM_10"] = close.diff(10)

    stoch_k = ((close - low.rolling(14).min()) /
               (high.rolling(14).max() - low.rolling(14).min() + 1e-9)) * 100
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_k.rolling(3).mean()

    # ── Volatility indicators ─────────────────────────────────────
    df["BB_mid"]   = close.rolling(20).mean()
    bb_std         = close.rolling(20).std()
    df["BB_upper"] = df["BB_mid"] + 2 * bb_std
    df["BB_lower"] = df["BB_mid"] - 2 * bb_std
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / (df["BB_mid"] + 1e-9)

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    df["ATR_14"] = tr.rolling(14).mean()

    # ── Volume indicators ─────────────────────────────────────────
    df["OBV"] = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["Vol_SMA_20"] = vol.rolling(20).mean()

    # ── Label: Buy / Hold / Sell ──────────────────────────────────
    future_return = close.shift(-5) / close - 1  # 5-day forward return
    df["Label"] = pd.cut(
        future_return,
        bins=[-np.inf, -0.02, 0.02, np.inf],
        labels=[0, 1, 2]          # 0=Sell, 1=Hold, 2=Buy
    ).astype(float)

    df.dropna(inplace=True)
    return df


def build_dataset(raw_csv="data/all_stocks.csv"):
    df = pd.read_csv(raw_csv, index_col=0, parse_dates=True)
    df.columns = df.columns.str.strip()

    frames = []
    for ticker, group in df.groupby("Ticker"):
        group = group.copy().sort_index()
        group = add_technical_indicators(group)
        frames.append(group)
        print(f"{ticker}: {len(group)} rows after feature engineering")

    full = pd.concat(frames).dropna()
    full.to_csv("data/featured_stocks.csv")
    print(f"\nFeatured dataset saved → data/featured_stocks.csv ({len(full)} rows)")
    return full


if __name__ == "__main__":
    build_dataset()