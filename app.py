from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import warnings
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

from src.predict import predict

app = FastAPI(
    title="Stock Recommendation API",
    description="Buy / Hold / Sell signals for NSE stocks using XGBoost",
    version="1.0.0"
)


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
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def get_prediction(request: PredictRequest):
    ticker = request.ticker.upper().strip()

    if not ticker:
        raise HTTPException(status_code=400, detail="Ticker cannot be empty")

    # Add .NS suffix automatically if user forgets
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
    """Convenience GET endpoint — /predict/TCS.NS"""
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