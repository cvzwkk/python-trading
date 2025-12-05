#!pip install python-binance
#!pip install binance
#!pip install pykalman

import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# Binance
from binance.client import Client

# ML Libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Connect to Binance US public data
client = Client(api_key="", api_secret="", tld='us')

# ==========================================================
# Fetch OHLCV Bars from Binance US
# ==========================================================
def fetch_ohlcv_bars(symbol="BTCUSDT", interval="1m", limit=60):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]
    df[["open","high","low","close","volume"]] = df[["open","high","low","close","volume"]].astype(float)
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
# Trend Signal
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

# ==========================================================
# Ridge Model Prediction
# ==========================================================
def ridge_prediction(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict([Xs[-1]])[0]

# ==========================================================
# Live Loop
# ==========================================================
symbol = "BTCUSDT"
interval = "1m"

print("Starting live Binance US data... (updates every 3 seconds)")

while True:
    try:
        bars = fetch_ohlcv_bars(symbol, interval, limit=60)
        df, X, y = prepare_features(bars)
        last_close = df["close"].iloc[-1]
        pred = ridge_prediction(X, y)
        signal = trend_signal(pred, last_close)
        
        print(f"[{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC] Last Close: {last_close:.2f} | Ridge Prediction: {pred:.2f} → {signal}")
        
        time.sleep(3)  # wait 3 seconds before fetching again
    except KeyboardInterrupt:
        print("Stopped by user")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(3)
