
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from binance.client import Client
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from pykalman import KalmanFilter

# Binance US Client (no key)
client = Client()

# ==========================================================
# Fetch OHLCV Bars from Binance
# ==========================================================
def fetch_ohlcv_bars(symbol="BTCUSDT", interval="1s", limit=50):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","qav","num_trades","taker_base","taker_quote","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["datetime","open","high","low","close","volume"]]

# ==========================================================
# VWAP + Std
# ==========================================================
def compute_vwap_std(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)
    vwap = (tp * vol).sum() / vol.sum()
    std1 = np.sqrt(((tp - vwap) ** 2).mean())
    return {
        "VWAP": vwap,
        "Upper 1σ": vwap + std1,
        "Lower 1σ": vwap - std1,
        "Latest close": df["close"].iloc[-1]
    }

# ==========================================================
# Feature Engineering
# ==========================================================
def prepare_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma5"] = df["close"].rolling(5).mean()
    df["ma10"] = df["close"].rolling(10).mean()
    df["volatility"] = df["return"].rolling(5).std()
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    features = ["close", "ma5", "ma10", "volatility", "volume"]
    X = df[features]
    y = df["target"]
    return df, X, y

# ==========================================================
# Trend Signal Mapping
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

# ==========================================================
# Aggregate all signals
# ==========================================================
def aggregate_trend_signals(*signals):
    mapping = {"BULLISH 📈": 1, "BEARISH 📉": -1, "NEUTRAL ➖": 0}
    total = sum([mapping.get(s,0) for s in signals])
    if total > 0:
        return f"BULLISH 📈 ({total})"
    elif total < 0:
        return f"BEARISH 📉 ({total})"
    else:
        return f"NEUTRAL ➖ ({total})"

# ==========================================================
# ML Model Examples (Fast)
# ==========================================================
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict([Xs[-1]])[0]

def model_rf(X, y):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]

def model_xgb(X, y):
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3)
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]

def apply_kalman(df):
    kf = KalmanFilter(initial_state_mean=df["close"].iloc[0], n_dim_obs=1, n_dim_state=1)
    smoothed, _ = kf.smooth(df["close"].values)
    return float(smoothed[-1])

def ema_signal(df, span=5):
    ema = df["close"].ewm(span=span).mean().iloc[-1]
    return "BULLISH 📈" if ema > df["close"].iloc[-1] else "BEARISH 📉"

# ==========================================================
# LIVE LOOP
# ==========================================================
symbol = "BTCUSDT"
interval = "1s"

while True:
    try:
        bars = fetch_ohlcv_bars(symbol=symbol, interval=interval, limit=50)
        last_close = bars["close"].iloc[-1]
        df, X, y = prepare_features(bars)
        summary = compute_vwap_std(bars)

        # Predictions
        ridge_pred = model_ridge(X, y)
        rf_pred = model_rf(X, y)
        xgb_pred = model_xgb(X, y)
        kalman_pred = apply_kalman(df)
        ema_sig = ema_signal(bars)

        # Signals
        ridge_sig = trend_signal(ridge_pred, last_close)
        rf_sig = trend_signal(rf_pred, last_close)
        xgb_sig = trend_signal(xgb_pred, last_close)
        kalman_sig = trend_signal(kalman_pred, last_close)

        # Aggregate
        final_signal = aggregate_trend_signals(
            ridge_sig, rf_sig, xgb_sig, kalman_sig, ema_sig
        )

        # Print live output
        print(f"\n{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Current Price: {last_close:.2f}")
        print(f"Ridge: {ridge_pred:.2f} → {ridge_sig}")
        print(f"Random Forest: {rf_pred:.2f} → {rf_sig}")
        print(f"XGBoost: {xgb_pred:.2f} → {xgb_sig}")
        print(f"Kalman: {kalman_pred:.2f} → {kalman_sig}")
        print(f"EMA Signal: → {ema_sig}")
        print(f"=== FINAL TREND SIGNAL: {final_signal} ===")

        time.sleep(1)
    except Exception as e:
        print("Error:", e)
        time.sleep(3)
