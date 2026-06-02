from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import warnings
import os
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

app = FastAPI(
    title="Stock Recommendation API",
    description="Buy / Hold / Sell signals for NSE stocks using XGBoost",
    version="1.0.0"
)

# Load model lazily so startup errors are visible
try:
    from src.predict import predict
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_ERROR  = str(e)


# ── Request / Response schemas ────────────────────────────────────────
class PredictRequest(BaseModel):
    ticker: str

class Probabilities(BaseModel):
    Sell: float
    Hold: float
    Buy:  float

class PredictResponse(BaseModel):
    ticker:        str
    signal:        str
    confidence:    float
    probabilities: Probabilities


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Stock Recommendation API is live!",
        "usage":   "POST /predict with body: {\"ticker\": \"RELIANCE.NS\"}"
    }


@app.get("/health")
def health():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_ERROR}")
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_ERROR}")

    ticker = request.ticker.upper().strip()
    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")
    if "." not in ticker:
        ticker = ticker + ".NS"

    try:
        result = predict(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        ticker=result["ticker"],
        signal=result["signal"],
        confidence=result["confidence"],
        probabilities=Probabilities(**result["probabilities"])
    )


@app.get("/predict/{ticker}")
def get_prediction_by_url(ticker: str):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {MODEL_ERROR}")

    ticker = ticker.upper().strip()
    if "." not in ticker:
        ticker = ticker + ".NS"

    try:
        result = predict(ticker)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return PredictResponse(
        ticker=result["ticker"],
        signal=result["signal"],
        confidence=result["confidence"],
        probabilities=Probabilities(**result["probabilities"])
    )