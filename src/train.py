import pandas as pd
import numpy as np
import joblib
import os
import optuna
import warnings
warnings.filterwarnings("ignore")

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Features used by the model ────────────────────────────────────────
FEATURES = [
    "SMA_20", "SMA_50", "EMA_20", "MACD", "MACD_signal",
    "RSI", "ROC_10", "MOM_10", "STOCH_K", "STOCH_D",
    "BB_mid", "BB_upper", "BB_lower", "BB_width",
    "ATR_14", "OBV", "Vol_SMA_20", "Open", "Volume"
]
TARGET = "Label"


def load_data(path="data/featured_stocks.csv"):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)
    print(f"Loaded {len(df)} rows | Classes: {df[TARGET].value_counts().to_dict()}")
    return df


def time_aware_split(df, train_ratio=0.7):
    """Split by time — no data leakage. Each ticker split independently."""
    train_frames, test_frames = [], []

    for ticker, group in df.groupby("Ticker"):
        group = group.sort_index()
        split = int(len(group) * train_ratio)
        train_frames.append(group.iloc[:split])
        test_frames.append(group.iloc[split:])

    train = pd.concat(train_frames)
    test  = pd.concat(test_frames)
    print(f"Train: {len(train)} rows | Test: {len(test)} rows")
    return train, test


def apply_smote(X_train, y_train):
    """Apply SMOTE only on training data to fix class imbalance."""
    print(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"After SMOTE:  {pd.Series(y_res).value_counts().to_dict()}")
    return X_res, y_res


def optimize_xgboost(X_train, y_train, n_trials=30):
    """Optuna hyperparameter tuning."""
    print(f"\nRunning Optuna ({n_trials} trials)...")

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 0.5),
            "use_label_encoder": False,
            "eval_metric":       "mlogloss",
            "random_state":      42,
            "n_jobs":            -1,
        }
        model = XGBClassifier(**params)

        # Quick cross-val on a 80/20 slice of training data
        split = int(len(X_train) * 0.8)
        model.fit(X_train[:split], y_train[:split], verbose=False)
        preds = model.predict(X_train[split:])
        return accuracy_score(y_train[split:], preds)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"Best accuracy (val): {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    return study.best_params


def train(data_path="data/featured_stocks.csv", n_trials=30):
    # 1. Load
    df = load_data(data_path)

    # 2. Time-aware split
    train_df, test_df = time_aware_split(df)

    X_train = train_df[FEATURES].values
    y_train = train_df[TARGET].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df[TARGET].values

    # 3. SMOTE on training data only
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # 4. Optuna tuning
    best_params = optimize_xgboost(X_train_res, y_train_res, n_trials=n_trials)

    # 5. Train final model with best params
    print("\nTraining final model...")
    best_params.update({
        "use_label_encoder": False,
        "eval_metric": "mlogloss",
        "random_state": 42,
        "n_jobs": -1,
    })
    model = XGBClassifier(**best_params)
    model.fit(X_train_res, y_train_res, verbose=False)

    # 6. Evaluate
    preds = model.predict(X_test)
    acc   = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Sell", "Hold", "Buy"]))

    # 7. Save model + feature list
    os.makedirs("models", exist_ok=True)
    joblib.dump(model,    "models/xgb_model.joblib")
    joblib.dump(FEATURES, "models/features.joblib")
    print("Model saved → models/xgb_model.joblib")

    return model


if __name__ == "__main__":
    train()