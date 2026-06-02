# Stock Recommendation System

An XGBoost-based Buy/Hold/Sell classifier for Indian NSE stocks.

## Tech Stack
- XGBoost · Optuna · SMOTE · Pandas · FastAPI

## Features
- 19 engineered technical indicators (trend, momentum, volatility)
- Time-aware 70/30 train/test split (no data leakage)
- SMOTE applied only on training data to handle class imbalance
- Optuna hyperparameter tuning (30 trials)
- REST API with FastAPI

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info |
| GET | `/health` | Health check |
| GET | `/predict/{ticker}` | Predict by URL |
| POST | `/predict` | Predict via JSON body |

## Usage

```bash
# GET
curl https://your-app.onrender.com/predict/TCS.NS

# POST
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"ticker": "RELIANCE.NS"}'
```

## Sample Response

```json
{
  "ticker": "TCS.NS",
  "signal": "Hold",
  "confidence": 67.56,
  "probabilities": {
    "Sell": 25.61,
    "Hold": 67.56,
    "Buy": 6.83
  }
}
```