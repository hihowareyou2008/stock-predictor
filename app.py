"""
Stock Price Prediction Backend
Flask API + scikit-learn ML models
"""

import json
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

app = Flask(__name__)

# ── Data Loading ──────────────────────────────────────────────────────────────

DATA_PATH = "stock_data.csv"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df.rename(columns={"Unnamed: 0": "Date"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df

DF = load_data()
STOCK_COLS = [c for c in DF.columns if c.startswith("Stock_")]


# ── Feature Engineering ───────────────────────────────────────────────────────

def build_features(prices: np.ndarray, window: int):
    """
    Build a supervised ML dataset from a price series.
    Features per row:
      - lag_1 .. lag_window   : past prices
      - rolling_mean          : mean of window
      - rolling_std           : std of window
      - momentum              : price[t-1] - price[t-window]
      - roc                   : rate of change (%)
    Target: price at time t
    """
    X, y = [], []
    for i in range(window, len(prices)):
        window_vals = prices[i - window: i]
        lag_feats   = list(window_vals)                          # lag_1..lag_N
        roll_mean   = float(np.mean(window_vals))
        roll_std    = float(np.std(window_vals))
        momentum    = float(prices[i - 1] - prices[i - window])
        roc         = float((prices[i - 1] - prices[i - window]) /
                            (prices[i - window] + 1e-9) * 100)
        X.append(lag_feats + [roll_mean, roll_std, momentum, roc])
        y.append(prices[i])
    return np.array(X), np.array(y)


def feature_names(window: int):
    names = [f"lag_{i}" for i in range(1, window + 1)]
    names += ["rolling_mean", "rolling_std", "momentum", "roc"]
    return names


# ── Model Factory ─────────────────────────────────────────────────────────────

def get_model(model_type: str):
    if model_type == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  LinearRegression())
        ])
    elif model_type == "poly":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=2, include_bias=False)),
            ("model",  LinearRegression())
        ])
    elif model_type == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            ))
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


# ── Feature Importance ────────────────────────────────────────────────────────

def get_feature_importance(pipeline, model_type: str, feat_names: list):
    model = pipeline.named_steps["model"]
    if model_type == "rf":
        importances = model.feature_importances_
    elif model_type in ("linear", "poly"):
        coefs = np.abs(model.coef_)
        # for poly the coef length differs from feat_names → normalise by index
        importances = coefs[:len(feat_names)]
        total = importances.sum() or 1
        importances = importances / total
    else:
        importances = np.ones(len(feat_names)) / len(feat_names)

    # align length just in case poly expanded features
    importances = importances[:len(feat_names)]
    total = importances.sum() or 1
    importances = importances / total

    pairs = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]


# ── API Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stocks", methods=["GET"])
def api_stocks():
    """Return stock names and their latest prices."""
    result = []
    for col in STOCK_COLS:
        latest = float(DF[col].iloc[-1])
        prev   = float(DF[col].iloc[-2])
        change = round((latest - prev) / prev * 100, 2)
        result.append({
            "name":   col,
            "latest": round(latest, 2),
            "change": change
        })
    return jsonify({"stocks": result})


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return full price history for a given stock."""
    stock = request.args.get("stock", STOCK_COLS[0])
    if stock not in STOCK_COLS:
        return jsonify({"error": "Unknown stock"}), 400

    dates  = DF["Date"].dt.strftime("%Y-%m-%d").tolist()
    prices = DF[stock].round(2).tolist()
    return jsonify({"dates": dates, "prices": prices, "stock": stock})


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Train a model and return predictions + metrics.

    Request JSON:
      stock      : str   e.g. "Stock_1"
      model      : str   "linear" | "poly" | "rf"
      window     : int   lookback window (days)
      split      : float train fraction 0.5–0.9
      horizon    : int   forecast days ahead
    """
    body = request.get_json(force=True)
    stock   = body.get("stock",   STOCK_COLS[0])
    model_t = body.get("model",   "linear")
    window  = int(body.get("window",  10))
    split   = float(body.get("split", 0.8))
    horizon = int(body.get("horizon", 7))

    if stock not in STOCK_COLS:
        return jsonify({"error": "Unknown stock"}), 400

    prices = DF[stock].values.astype(float)
    dates  = DF["Date"].dt.strftime("%Y-%m-%d").tolist()

    # Build feature matrix
    X, y = build_features(prices, window)
    feat_dates = dates[window:]           # dates aligned with rows in X/y

    # Train / test split
    n_train = max(int(len(X) * split), window + 1)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    dates_test = feat_dates[n_train:]

    # Fit
    model = get_model(model_t)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred)

    # Residuals
    residuals = (y_test - y_pred).tolist()

    # Feature importance
    fnames = feature_names(window)
    importance = get_feature_importance(model, model_t, fnames)

    # ── Forecast future N days ───────────────────────────────────────────────
    forecast_prices = []
    # Start rolling window from the last `window` known prices
    rolling = list(prices[-window:])

    for _ in range(horizon):
        arr = np.array(rolling[-window:])
        roll_mean = float(np.mean(arr))
        roll_std  = float(np.std(arr))
        momentum  = float(arr[-1] - arr[0])
        roc       = float((arr[-1] - arr[0]) / (arr[0] + 1e-9) * 100)
        feat_row  = list(arr) + [roll_mean, roll_std, momentum, roc]
        next_p    = float(model.predict([feat_row])[0])
        forecast_prices.append(round(next_p, 2))
        rolling.append(next_p)

    # Confidence interval: ±1.96 * residual std
    resid_std = float(np.std(residuals)) if residuals else 1.0
    ci_low  = [round(p - 1.96 * resid_std, 2) for p in forecast_prices]
    ci_high = [round(p + 1.96 * resid_std, 2) for p in forecast_prices]

    # Generate forecast date labels
    last_date = DF["Date"].iloc[-1]
    forecast_dates = [
        (last_date + pd.Timedelta(days=i + 1)).strftime("%Y-%m-%d")
        for i in range(horizon)
    ]

    return jsonify({
        "stock":   stock,
        "model":   model_t,
        "window":  window,
        "split":   split,
        "metrics": metrics,
        "test": {
            "dates":     dates_test,
            "actual":    [round(float(v), 2) for v in y_test],
            "predicted": [round(float(v), 2) for v in y_pred],
            "residuals": [round(float(v), 4) for v in residuals],
        },
        "train": {
            "dates":  feat_dates[:n_train],
            "actual": [round(float(v), 2) for v in y_train],
        },
        "forecast": {
            "dates":  forecast_dates,
            "prices": forecast_prices,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "feature_importance": importance,
    })


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Stock Predictor API on http://localhost:5000")
    app.run(host="0.0.0.0", port=8080)
