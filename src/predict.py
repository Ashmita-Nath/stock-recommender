import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

import yfinance as yf

model    = joblib.load("models/xgb_model.joblib")
FEATURES = joblib.load("models/features.joblib")

LABEL_MAP = {0: "Sell", 1: "Hold", 2: "Buy"}


def get_latest_features(ticker: str) -> pd.DataFrame:
    """Download recent data and compute indicators for one ticker."""
    df = yf.download(ticker, period="6mo", auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    df["SMA_20"]      = close.rolling(20).mean()
    df["SMA_50"]      = close.rolling(50).mean()
    df["EMA_20"]      = close.ewm(span=20, adjust=False).mean()
    df["MACD"]        = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    df["MACD_signal"] = df["MACD"].ewm(span=9).mean()

    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    df["ROC_10"]  = close.pct_change(10) * 100
    df["MOM_10"]  = close.diff(10)

    stoch_k = ((close - low.rolling(14).min()) /
               (high.rolling(14).max() - low.rolling(14).min() + 1e-9)) * 100
    df["STOCH_K"] = stoch_k
    df["STOCH_D"] = stoch_k.rolling(3).mean()

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

    df["OBV"]        = (np.sign(close.diff()) * vol).fillna(0).cumsum()
    df["Vol_SMA_20"] = vol.rolling(20).mean()

    df.dropna(inplace=True)

    if df.empty:
        raise ValueError(f"Not enough data to compute indicators for {ticker}")

    return df[FEATURES].iloc[[-1]]   # return only the latest row


def predict(ticker: str) -> dict:
    features    = get_latest_features(ticker)
    pred_class  = int(model.predict(features)[0])
    proba       = model.predict_proba(features)[0]

    return {
        "ticker":     ticker,
        "signal":     LABEL_MAP[pred_class],
        "confidence": round(float(proba[pred_class]) * 100, 2),
        "probabilities": {
            "Sell": round(float(proba[0]) * 100, 2),
            "Hold": round(float(proba[1]) * 100, 2),
            "Buy":  round(float(proba[2]) * 100, 2),
        }
    }


if __name__ == "__main__":
    result = predict("TCS.NS")
    print(result)