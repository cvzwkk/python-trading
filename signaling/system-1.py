
# Install Streamlit if needed
# !pip install streamlit

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time

# ML Libraries
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# Deep Learning
import tensorflow as tf
from tensorflow.keras import layers, models

# Kalman
from pykalman import KalmanFilter

# Streamlit
import streamlit as st

# ==========================================================
# Fetch OHLCV Bars
# ==========================================================
def fetch_ohlcv_bars(exchange_id="binanceus", symbol="BTC/USDT", timeframe="4h", limit=1000):
    ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})
    ex.load_markets()
    ohlcv = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)

    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

    return df

# ==========================================================
# VWAP + Standard Deviation Bands
# ==========================================================
def compute_vwap_std(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)

    vwap = (tp * vol).sum() / vol.sum()
    std1 = np.sqrt(((tp - vwap) ** 2).mean())
    std2 = std1 * 2

    return {
        "VWAP": vwap,
        "Upper 1σ": vwap + std1,
        "Upper 2σ": vwap + std2,
        "Lower 1σ": vwap - std1,
        "Lower 2σ": vwap - std2,
        "Latest datetime": df["datetime"].iloc[-1],
        "Latest close": df["close"].iloc[-1]
    }

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
# ML Models
# ==========================================================
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict([Xs[-1]])[0]

def model_xgboost(X, y):
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5)
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]

def model_gbm(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]

# ==========================================================
# Kalman Filter
# ==========================================================
def apply_kalman(df):
    kf = KalmanFilter(initial_state_mean=df["close"].iloc[0], n_dim_obs=1, n_dim_state=1)
    smoothed, _ = kf.smooth(df["close"].values)
    return float(smoothed[-1])

# ==========================================================
# Autoencoder
# ==========================================================
def model_autoencoder(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    input_dim = Xs.shape[1]
    encoder = models.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='linear', name="latent")
    ])
    decoder = models.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(Xs, Xs, epochs=20, verbose=0)
    latent_vec = encoder.predict(Xs[-1:])
    return latent_vec[0]

# ==========================================================
# GRU Model
# ==========================================================
def model_gru(df):
    seq = 30
    data = df["close"].values
    X, Y = [], []
    for i in range(len(data)-seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])
    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)
    model = models.Sequential([layers.GRU(32), layers.Dense(1)])
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)
    return model.predict(X[-1].reshape(1, seq, 1))[0][0]

# ==========================================================
# Transformer Model
# ==========================================================
def model_transformer(df):
    seq = 30
    data = df["close"].values
    X, Y = [], []
    for i in range(len(data)-seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])
    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)
    inp = layers.Input(shape=(seq, 1))
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inp, inp)
    x = layers.LayerNormalization()(inp + attn)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)
    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)
    return model.predict(X[-1].reshape(1, seq, 1))[0][0]

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
# Streamlit Dashboard
# ==========================================================
st.title("Live BTC/USDT Dashboard")
placeholder = st.empty()

symbol = "BTC/USDT"
exchange_id = "binanceus"

while True:
    try:
        bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe="1w", limit=224)
        summary = compute_vwap_std(bars)
        df, X, y = prepare_features(bars)

        ridge_pred = model_ridge(X, y)
        xgb_pred = model_xgboost(X, y)
        gbm_pred = model_gbm(X, y)
        kalman_pred = apply_kalman(df)
        ae_latent = model_autoencoder(X)
        gru_pred = model_gru(df)
        transformer_pred = model_transformer(df)
        last_close = df["close"].iloc[-1]

        ridge_signal = trend_signal(ridge_pred, last_close)
        xgb_signal = trend_signal(xgb_pred, last_close)
        gbm_signal = trend_signal(gbm_pred, last_close)
        kalman_signal = trend_signal(kalman_pred, last_close)
        gru_signal = trend_signal(gru_pred, last_close)
        transformer_signal = trend_signal(transformer_pred, last_close)

        # Build text output
        output = f"\nUpdate at: {datetime.now(timezone.utc)} UTC\n"
        output += "=== VWAP Summary ===\n"
        for k, v in summary.items():
            output += f"{k:15}: {v}\n"
        output += "\n=== Machine Learning Forecasts ===\n"
        output += f"Current Close:        {last_close:.2f}\n\n"
        output += f"Ridge Regression:     {ridge_pred:.2f}   → {ridge_signal}\n"
        output += f"XGBoost:              {xgb_pred:.2f}   → {xgb_signal}\n"
        output += f"GBM:                  {gbm_pred:.2f}   → {gbm_signal}\n"
        output += f"Kalman Smoothed:      {kalman_pred:.2f}   → {kalman_signal}\n"
        output += f"GRU Forecast:         {gru_pred:.2f}   → {gru_signal}\n"
        output += f"Transformer Forecast: {transformer_pred:.2f}   → {transformer_signal}\n"
        output += "\nAutoencoder latent factors: " + str(ae_latent)

        # Update dashboard
        placeholder.text(output)
        print(output)  # keep console history

        time.sleep(3)

    except KeyboardInterrupt:
        print("Live update stopped by user.")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(3)# ==========================================================
# VWAP + Standard Deviation Bands
# ==========================================================
def compute_vwap_std(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].fillna(0)

    vwap = (tp * vol).sum() / vol.sum()
    std1 = np.sqrt(((tp - vwap) ** 2).mean())
    std2 = std1 * 2

    return {
        "VWAP": vwap,
        "Upper 1σ": vwap + std1,
        "Upper 2σ": vwap + std2,
        "Lower 1σ": vwap - std1,
        "Lower 2σ": vwap - std2,
        "Latest datetime": df["datetime"].iloc[-1],
        "Latest close": df["close"].iloc[-1]
    }


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
# ML Models
# ==========================================================
def model_ridge(X, y):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, _, y_train, _ = train_test_split(Xs, y, test_size=0.2)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model.predict([Xs[-1]])[0]


def model_xgboost(X, y):
    model = XGBRegressor(n_estimators=300, learning_rate=0.03, max_depth=5)
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]


def model_gbm(X, y):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    return model.predict([X.iloc[-1]])[0]


# ==========================================================
# Kalman Filter (Smoothed last value)
# ==========================================================
def apply_kalman(df):
    kf = KalmanFilter(
        initial_state_mean=df["close"].iloc[0],
        n_dim_obs=1,
        n_dim_state=1
    )
    smoothed, _ = kf.smooth(df["close"].values)
    return float(smoothed[-1])


# ==========================================================
# Autoencoder (Latent Factors)
# ==========================================================
def model_autoencoder(X):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    input_dim = Xs.shape[1]

    encoder = models.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='linear', name="latent")
    ])

    decoder = models.Sequential([
        layers.Dense(4, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])

    autoencoder = models.Sequential([encoder, decoder])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(Xs, Xs, epochs=20, verbose=0)

    latent_vec = encoder.predict(Xs[-1:])
    return latent_vec[0]


# ==========================================================
# GRU Model
# ==========================================================
def model_gru(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i+seq])
        Y.append(data[i+seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    model = models.Sequential([
        layers.GRU(32),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return model.predict(X[-1].reshape(1, seq, 1))[0][0]


# ==========================================================
# Transformer
# ==========================================================
def model_transformer(df):
    seq = 30
    data = df["close"].values

    X, Y = [], []
    for i in range(len(data) - seq):
        X.append(data[i:i + seq])
        Y.append(data[i + seq])

    X = np.array(X).reshape(-1, seq, 1)
    Y = np.array(Y)

    inp = layers.Input(shape=(seq, 1))
    attn = layers.MultiHeadAttention(num_heads=2, key_dim=16)(inp, inp)
    x = layers.LayerNormalization()(inp + attn)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)

    model = models.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")
    model.fit(X, Y, epochs=10, verbose=0)

    return model.predict(X[-1].reshape(1, seq, 1))[0][0]


# ==========================================================
# Trend Signal Function
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"


# ==========================================================
# MAIN EXECUTION WITH LIVE UPDATES
# ==========================================================
symbol = "BTC/USDT"
exchange_id = "binanceus"

# Approximate number of lines your output occupies
OUTPUT_LINES = 25

while True:
    try:
        # Fetch latest bars
        bars = fetch_ohlcv_bars(exchange_id, symbol, timeframe="1w", limit=224)
        summary = compute_vwap_std(bars)
        df, X, y = prepare_features(bars)

        # Predictions
        ridge_pred = model_ridge(X, y)
        xgb_pred = model_xgboost(X, y)
        gbm_pred = model_gbm(X, y)
        kalman_pred = apply_kalman(df)
        ae_latent = model_autoencoder(X)
        gru_pred = model_gru(df)
        transformer_pred = model_transformer(df)

        last_close = df["close"].iloc[-1]

        # Signals
        ridge_signal = trend_signal(ridge_pred, last_close)
        xgb_signal = trend_signal(xgb_pred, last_close)
        gbm_signal = trend_signal(gbm_pred, last_close)
        kalman_signal = trend_signal(kalman_pred, last_close)
        gru_signal = trend_signal(gru_pred, last_close)
        transformer_signal = trend_signal(transformer_pred, last_close)

        # Move cursor up to overwrite previous output
        sys.stdout.write(f"\033[{OUTPUT_LINES}A")  # Move cursor up
        sys.stdout.flush()

    except KeyboardInterrupt:
        print("\nLive update stopped by user.")
        break
    except Exception as e:
        print(f"\nError occurred: {e}")
        time.sleep(3)
        continue

    # Print output
    print("\n==============================================")
    print(f"Update at: {datetime.now(timezone.utc)} UTC")
    print("=== VWAP Summary ===")
    for k, v in summary.items():
        print(f"{k:15}: {v}")

    print("\n=== Machine Learning Forecasts ===")
    print(f"Current Close:        {last_close:.2f}\n")
    print(f"Ridge Regression:     {ridge_pred:.2f}   → {ridge_signal}")
    print(f"XGBoost:              {xgb_pred:.2f}   → {xgb_signal}")
    print(f"GBM:                  {gbm_pred:.2f}   → {gbm_signal}")
    print(f"Kalman Smoothed:      {kalman_pred:.2f}   → {kalman_signal}")
    print(f"GRU Forecast:         {gru_pred:.2f}   → {gru_signal}")
    print(f"Transformer Forecast: {transformer_pred:.2f}   → {transformer_signal}")
    print("\nAutoencoder latent factors:", ae_latent)

    # Wait 3 seconds before next update
    time.sleep(3)
