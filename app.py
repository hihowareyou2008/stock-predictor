"""
Stock Price Prediction Backend
Flask API + scikit-learn ML models
"""

import os
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

app = Flask(__name__)


def resolve_path(env_var: str, *candidates: Path):
    env_value = os.getenv(env_var)
    if env_value:
        return Path(env_value)
    for candidate in candidates:
        if Path(candidate).exists():
            return Path(candidate)
    return Path(candidates[0])


BASE_DIR = Path(__file__).resolve().parent
ZIP_PATH = resolve_path(
    "STOCKS_ZIP_PATH",
    BASE_DIR / "stocks.zip",
    Path(r"C:\Users\praja\Downloads\stocks.zip"),
)
METADATA_PATH = resolve_path(
    "SYMBOLS_META_PATH",
    BASE_DIR / "symbols_valid_meta.csv",
    Path(r"C:\Users\praja\Downloads\symbols_valid_meta.csv"),
)
DEFAULT_LIMIT = 60


def validate_data_sources():
    missing = []
    if not ZIP_PATH.exists():
        missing.append(f"stocks zip not found at {ZIP_PATH}")
    if not METADATA_PATH.exists():
        missing.append(f"metadata csv not found at {METADATA_PATH}")
    if missing:
        raise FileNotFoundError(" | ".join(missing))


def load_symbol_metadata():
    df = pd.read_csv(METADATA_PATH)
    df = df[["Symbol", "Security Name"]].dropna()
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Security Name"] = df["Security Name"].astype(str).str.strip()
    return {
        row["Symbol"]: row["Security Name"]
        for _, row in df.iterrows()
    }


def load_available_tickers():
    with ZipFile(ZIP_PATH) as archive:
        tickers = []
        for entry in archive.infolist():
            if not entry.filename.lower().endswith(".csv"):
                continue
            ticker = Path(entry.filename).stem.upper()
            if ticker:
                tickers.append(ticker)
    return sorted(set(tickers))


validate_data_sources()
SYMBOL_TO_NAME = load_symbol_metadata()
AVAILABLE_TICKERS = load_available_tickers()
STOCK_COLS = AVAILABLE_TICKERS


def stock_meta_for_ticker(ticker: str):
    security_name = SYMBOL_TO_NAME.get(ticker, ticker)
    return {
        "series_key": ticker,
        "ticker": ticker,
        "security_name": security_name,
        "display_name": f"{ticker} - {security_name}",
    }


@lru_cache(maxsize=512)
def load_stock_frame(ticker: str):
    if ticker not in STOCK_COLS:
        raise KeyError(ticker)

    with ZipFile(ZIP_PATH) as archive:
        with archive.open(f"{ticker}.csv") as handle:
            df = pd.read_csv(handle, usecols=["Date", "Adj Close"])

    df["Date"] = pd.to_datetime(df["Date"])
    df["Adj Close"] = pd.to_numeric(df["Adj Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Adj Close"]).sort_values("Date").reset_index(drop=True)
    return df


@lru_cache(maxsize=2048)
def stock_summary(ticker: str):
    df = load_stock_frame(ticker)
    latest = float(df["Adj Close"].iloc[-1])
    prev = float(df["Adj Close"].iloc[-2]) if len(df) > 1 else latest
    change = round((latest - prev) / (prev + 1e-9) * 100, 2)
    meta = stock_meta_for_ticker(ticker)
    return {
        "name": meta["ticker"],
        "ticker": meta["ticker"],
        "security_name": meta["security_name"],
        "series_key": meta["series_key"],
        "display_name": meta["display_name"],
        "latest": round(latest, 2),
        "change": change,
    }


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
    features, targets = [], []
    for index in range(window, len(prices)):
        window_vals = prices[index - window:index]
        lag_feats = list(window_vals)
        roll_mean = float(np.mean(window_vals))
        roll_std = float(np.std(window_vals))
        momentum = float(prices[index - 1] - prices[index - window])
        roc = float((prices[index - 1] - prices[index - window]) / (prices[index - window] + 1e-9) * 100)
        features.append(lag_feats + [roll_mean, roll_std, momentum, roc])
        targets.append(prices[index])
    return np.array(features), np.array(targets)


def feature_names(window: int):
    names = [f"lag_{i}" for i in range(1, window + 1)]
    names += ["rolling_mean", "rolling_std", "momentum", "roc"]
    return names


def get_model(model_type: str):
    if model_type == "linear":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ])
    if model_type == "poly":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("model", LinearRegression())
        ])
    if model_type == "rf":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=1
            ))
        ])
    raise ValueError(f"Unknown model type: {model_type}")


def compute_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100)
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def get_feature_importance(pipeline, model_type: str, feat_names: list):
    model = pipeline.named_steps["model"]
    if model_type == "rf":
        importances = model.feature_importances_
    elif model_type in ("linear", "poly"):
        coefs = np.abs(model.coef_)
        importances = coefs[:len(feat_names)]
        total = importances.sum() or 1
        importances = importances / total
    else:
        importances = np.ones(len(feat_names)) / len(feat_names)

    importances = importances[:len(feat_names)]
    total = importances.sum() or 1
    importances = importances / total

    pairs = sorted(zip(feat_names, importances), key=lambda item: item[1], reverse=True)
    return [{"feature": feature, "importance": round(float(value), 4)} for feature, value in pairs]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stocks", methods=["GET"])
def api_stocks():
    """Return searchable stock names and latest prices."""
    query = request.args.get("q", "").strip().lower()
    try:
        limit = max(1, min(int(request.args.get("limit", DEFAULT_LIMIT)), 200))
    except ValueError:
        limit = DEFAULT_LIMIT

    filtered = []
    for ticker in STOCK_COLS:
        meta = stock_meta_for_ticker(ticker)
        haystack = f"{ticker} {meta['security_name']}".lower()
        if query and query not in haystack:
            continue
        filtered.append(ticker)

    filtered = filtered[:limit]
    stocks = [stock_summary(ticker) for ticker in filtered]
    return jsonify({
        "stocks": stocks,
        "query": query,
        "count": len(stocks),
        "total_matches": len([ticker for ticker in STOCK_COLS if (not query) or query in f'{ticker} {SYMBOL_TO_NAME.get(ticker, ticker)}'.lower()]),
        "total_available": len(STOCK_COLS),
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    """Return full adjusted-close history for a given stock."""
    stock = request.args.get("stock", "")
    if stock not in STOCK_COLS:
        return jsonify({"error": "Unknown stock"}), 400

    meta = stock_meta_for_ticker(stock)
    df = load_stock_frame(stock)
    return jsonify({
        "dates": df["Date"].dt.strftime("%Y-%m-%d").tolist(),
        "prices": df["Adj Close"].round(2).tolist(),
        "stock": stock,
        "ticker": meta["ticker"],
        "security_name": meta["security_name"],
        "display_name": meta["display_name"],
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Train a model and return predictions + metrics.

    Request JSON:
      stock      : str   e.g. "AAPL"
      model      : str   "linear" | "poly" | "rf"
      window     : int   lookback window (days)
      split      : float train fraction 0.5-0.9
      horizon    : int   forecast days ahead
    """
    body = request.get_json(force=True)
    stock = body.get("stock", "")
    model_t = body.get("model", "linear")
    window = int(body.get("window", 10))
    split = float(body.get("split", 0.8))
    horizon = int(body.get("horizon", 7))

    if stock not in STOCK_COLS:
        return jsonify({"error": "Unknown stock"}), 400

    meta = stock_meta_for_ticker(stock)
    df = load_stock_frame(stock)
    prices = df["Adj Close"].values.astype(float)
    dates = df["Date"].dt.strftime("%Y-%m-%d").tolist()

    if len(prices) <= window + 2:
        return jsonify({"error": "Not enough data for the selected lookback window."}), 400

    features, targets = build_features(prices, window)
    feature_dates = dates[window:]

    n_train = max(int(len(features) * split), window + 1)
    n_train = min(n_train, len(features) - 1)
    x_train, x_test = features[:n_train], features[n_train:]
    y_train, y_test = targets[:n_train], targets[n_train:]
    dates_test = feature_dates[n_train:]

    model = get_model(model_t)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    metrics = compute_metrics(y_test, y_pred)
    residuals = (y_test - y_pred).tolist()
    importance = get_feature_importance(model, model_t, feature_names(window))

    forecast_prices = []
    rolling = list(prices[-window:])

    for _ in range(horizon):
        arr = np.array(rolling[-window:])
        roll_mean = float(np.mean(arr))
        roll_std = float(np.std(arr))
        momentum = float(arr[-1] - arr[0])
        roc = float((arr[-1] - arr[0]) / (arr[0] + 1e-9) * 100)
        feat_row = list(arr) + [roll_mean, roll_std, momentum, roc]
        next_price = float(model.predict([feat_row])[0])
        forecast_prices.append(round(next_price, 2))
        rolling.append(next_price)

    resid_std = float(np.std(residuals)) if residuals else 1.0
    ci_low = [round(price - 1.96 * resid_std, 2) for price in forecast_prices]
    ci_high = [round(price + 1.96 * resid_std, 2) for price in forecast_prices]

    last_date = df["Date"].iloc[-1]
    forecast_dates = [
        (last_date + pd.Timedelta(days=index + 1)).strftime("%Y-%m-%d")
        for index in range(horizon)
    ]

    return jsonify({
        "stock": stock,
        "ticker": meta["ticker"],
        "security_name": meta["security_name"],
        "display_name": meta["display_name"],
        "model": model_t,
        "window": window,
        "split": split,
        "metrics": metrics,
        "test": {
            "dates": dates_test,
            "actual": [round(float(value), 2) for value in y_test],
            "predicted": [round(float(value), 2) for value in y_pred],
            "residuals": [round(float(value), 4) for value in residuals],
        },
        "train": {
            "dates": feature_dates[:n_train],
            "actual": [round(float(value), 2) for value in y_train],
        },
        "forecast": {
            "dates": forecast_dates,
            "prices": forecast_prices,
            "ci_low": ci_low,
            "ci_high": ci_high,
        },
        "feature_importance": importance,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"Starting Stock Predictor API on http://localhost:{port}")
    print(f"Using stocks zip: {ZIP_PATH}")
    print(f"Using symbols meta: {METADATA_PATH}")
    app.run(host="0.0.0.0", port=port)
