
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone

from binance.client import Client

# ML Libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models

# Kalman Filter
from pykalman import KalmanFilter

# --------------------------
# Binance public client
# --------------------------
client = Client(api_key="", api_secret="")  # No key needed for public data

symbol = "BTCUSDT"
interval = "1m"
fetch_limit = 100  # enough for ma50 rolling

# ==========================================================
# Fetch OHLCV Bars from Binance
# ==========================================================
def fetch_ohlcv_bars(symbol, interval="1m", limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].astype(float)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

# ==========================================================
# Feature Engineering
# ==========================================================
def prepare_features(df):
    df = df.copy()
    df["return"] = df["close"].pct_change()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["volatility"] = df["return"].rolling(20).std()
    df["target"] = df["close"].shift(-1)
    df = df.dropna()
    features = ["close", "ma20", "ma50", "volatility", "volume"]
    X = df[features]
    y = df["target"]
    return df, X, y

# ==========================================================
# Trend signal
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

# ==========================================================
# Ridge model
# ==========================================================
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict([Xs[-1]])[0]

# ==========================================================
# Kalman Filter
# ==========================================================
def apply_kalman(df):
    kf = KalmanFilter(initial_state_mean=df["close"].iloc[0], n_dim_obs=1, n_dim_state=1)
    smoothed, _ = kf.smooth(df["close"].values)
    return float(smoothed[-1])

# ==========================================================
# Main live loop
# ==========================================================
while True:
    try:
        bars = fetch_ohlcv_bars(symbol, interval, fetch_limit)
        df, X, y = prepare_features(bars)

        if df.empty:
            print(f"{datetime.now(timezone.utc)} - Not enough data yet...")
            time.sleep(3)
            continue

        last_close = df["close"].iloc[-1]

        # ML Predictions
        ridge_pred = model_ridge(X, y)
        kalman_pred = apply_kalman(df)

        ridge_signal_val = trend_signal(ridge_pred, last_close)
        kalman_signal_val = trend_signal(kalman_pred, last_close)

        # Print live data
        print("\n=== Live BTC/USDT Update ===", datetime.now(timezone.utc))
        print(df[["datetime", "close", "volume"]].tail(10))  # last 10 prices
        print(f"\nLast Close: {last_close:.2f}")
        print(f"Ridge Prediction: {ridge_pred:.2f} → {ridge_signal_val}")
        print(f"Kalman Smoothed:  {kalman_pred:.2f} → {kalman_signal_val}")

    except Exception as e:
        print("Error:", e)

    time.sleep(1)  # update every 3 seconds
