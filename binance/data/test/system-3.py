

# -----------------------------
# Install needed packages
# -----------------------------
!pip install python-binance river

# -----------------------------
# Imports
# -----------------------------
from binance.client import Client
import pandas as pd
import numpy as np
import time
from datetime import datetime, timezone
from river import linear_model, preprocessing

# -----------------------------
# Binance US (No API Key needed for public data)
# -----------------------------
client = Client(tld="us",api_key="", api_secret="")  # empty keys for public endpoints

# -----------------------------
# Streaming predictor setup
# -----------------------------
# Online linear regression (incremental)
model = preprocessing.StandardScaler() | linear_model.LinearRegression()

# Parameters
symbol = "BTCUSDT"
window = 10  # last 10 seconds
trend_threshold = 0.0  # slope threshold for trend

# Storage
prices = []

# -----------------------------
# Feature extraction
# -----------------------------
def compute_features(prices):
    """
    Fast features for online prediction
    """
    arr = np.array(prices)
    features = {
        "slope": (arr[-1] - arr[0]) / len(arr),
        "momentum": arr[-1] - arr.mean(),
        "volatility": arr.std()
    }
    return features

# -----------------------------
# Trend signal
# -----------------------------
def trend_signal(pred_slope):
    if pred_slope > trend_threshold:
        return "BULLISH 📈"
    elif pred_slope < -trend_threshold:
        return "BEARISH 📉"
    else:
        return "NEUTRAL ➖"

# -----------------------------
# Live loop
# -----------------------------
print("Starting live trend predictor...")
while True:
    try:
        # Fetch latest price (1-second)
        tick = client.get_symbol_ticker(symbol=symbol)
        price = float(tick["price"])
        prices.append(price)
        if len(prices) > window:
            prices.pop(0)
        
        # Only compute if enough data
        if len(prices) >= window:
            # Compute features
            feats = compute_features(prices)
            X = {k: feats[k] for k in feats}
            
            # Train online model on latest price difference
            # target = next tick price (simulate by last delta)
            y = prices[-1] - prices[-2] if len(prices) > 1 else 0.0
            model.learn_one(X, y)
            
            # Predict future delta
            pred_delta = model.predict_one(X)
            
            # Determine trend
            trend = trend_signal(pred_delta)
            
            # Print live info
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Price: {price:.2f} → Trend: {trend} | Predicted delta: {pred_delta:.5f}")
        
        # Wait 1s
        time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stopping live predictor.")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(1)# Feature extraction
# -----------------------------
def compute_features(prices):
    """
    Fast features for online prediction
    """
    arr = np.array(prices)
    features = {
        "slope": (arr[-1] - arr[0]) / len(arr),
        "momentum": arr[-1] - arr.mean(),
        "volatility": arr.std()
    }
    return features

# -----------------------------
# Trend signal
# -----------------------------
def trend_signal(pred_slope):
    if pred_slope > trend_threshold:
        return "BULLISH 📈"
    elif pred_slope < -trend_threshold:
        return "BEARISH 📉"
    else:
        return "NEUTRAL ➖"

# -----------------------------
# Live loop
# -----------------------------
print("Starting live trend predictor...")
while True:
    try:
        # Fetch latest price (1-second)
        tick = client.get_symbol_ticker(symbol=symbol)
        price = float(tick["price"])
        prices.append(price)
        if len(prices) > window:
            prices.pop(0)
        
        # Only compute if enough data
        if len(prices) >= window:
            # Compute features
            feats = compute_features(prices)
            X = {k: feats[k] for k in feats}
            
            # Train online model on latest price difference
            # target = next tick price (simulate by last delta)
            y = prices[-1] - prices[-2] if len(prices) > 1 else 0.0
            model.learn_one(X, y)
            
            # Predict future delta
            pred_delta = model.predict_one(X)
            
            # Determine trend
            trend = trend_signal(pred_delta)
            
            # Print live info
            print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Price: {price:.2f} → Trend: {trend} | Predicted delta: {pred_delta:.5f}")
        
        # Wait 1s
        time.sleep(1)
    
    except KeyboardInterrupt:
        print("Stopping live predictor.")
        break
    except Exception as e:
        print("Error:", e)
        time.sleep(1)
