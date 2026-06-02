import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
import warnings
warnings.filterwarnings("ignore")

FEATURES = joblib.load("models/features.joblib")
TARGET   = "Label"


def load_test_data(path="data/featured_stocks.csv", train_ratio=0.7):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.dropna(subset=FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)

    test_frames = []
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_index()
        split = int(len(group) * train_ratio)
        test_frames.append(group.iloc[split:])

    return pd.concat(test_frames)


def evaluate():
    # 1. Load model + test data
    model   = joblib.load("models/xgb_model.joblib")
    test_df = load_test_data()

    X_test = test_df[FEATURES].values
    y_test = test_df[TARGET].values
    preds  = model.predict(X_test)

    # 2. Accuracy + report
    acc = accuracy_score(y_test, preds)
    print(f"Test Accuracy: {acc * 100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, preds,
          target_names=["Sell (0)", "Hold (1)", "Buy (2)"]))

    # 3. Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sell", "Hold", "Buy"])
    disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Confusion Matrix", fontsize=13)

    # 4. Feature importance
    importance = pd.Series(model.feature_importances_, index=FEATURES)
    importance = importance.sort_values(ascending=True)

    importance.plot(kind="barh", ax=axes[1], color="steelblue")
    axes[1].set_title("Feature Importance (XGBoost)", fontsize=13)
    axes[1].set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig("models/evaluation.png", dpi=150, bbox_inches="tight")
    print("Evaluation chart saved → models/evaluation.png")
    plt.show()

    # 5. Class distribution in test set
    print("\nClass distribution in test set:")
    counts = pd.Series(y_test).value_counts().sort_index()
    counts.index = ["Sell", "Hold", "Buy"]
    print(counts.to_string())


if __name__ == "__main__":
    evaluate()