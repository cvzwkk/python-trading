

!pip install python-binance pykalman --quiet

from binance.client import Client
from pykalman import KalmanFilter
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import time

# --- Parameters ---
symbol = "BTCUSDT"
interval = 1  # seconds
window_size = 30
price_history = []

# Binance public client
client = Client(tld="us", api_key="", api_secret="")

# ==========================================================
# Utility Functions
# ==========================================================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

def fetch_last_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

# ==========================================================
# Prediction Modes
# ==========================================================
def predict_lr(prices):
    if len(prices) < 2:
        return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m * len(prices) + c)

def predict_hma(prices, period=16):
    """
    Hull Moving Average prediction for the latest price.
    Returns the last price if not enough data is available.
    """
    if len(prices) < period:
        return prices[-1]

    def wma(arr, n):
        if len(arr) < n:
            return arr[-1]
        weights = np.arange(1, n + 1)
        return np.sum(arr[-n:] * weights) / weights.sum()

    half_length = period // 2
    sqrt_length = int(np.sqrt(period))

    wma_half = wma(np.array(prices), half_length)
    wma_full = wma(np.array(prices), period)

    raw_hma = 2 * wma_half - wma_full

    # Smooth final HMA
    final_hma = wma(np.array([raw_hma]), sqrt_length)
    return float(final_hma)


def predict_kalman(prices):
    if len(prices) < 2:
        return prices[-1]
    kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)
    state_means, _ = kf.smooth(np.array(prices))
    return float(state_means[-1])

def predict_cwma(prices):
    """
    Covariance-weighted moving average.
    Safe for small arrays and 1D data.
    """
    if len(prices) < 2:
        return prices[-1]

    returns = np.diff(prices)
    cov = np.cov(returns) if len(returns) > 1 else 1.0

    # weights scalar for 1D
    weight = 1 / (1 + cov)
    weighted_avg = np.average(prices, weights=np.full(len(prices), weight))
    return float(weighted_avg)


def predict_dma(prices, displacement=3):
    if len(prices) <= displacement:
        return prices[-1]
    return np.mean(prices[-displacement:])

# ==========================================================
# Final merged signal
# ==========================================================
def merge_signals(preds, last_price):
    # Each prediction gives +1 for bullish, -1 for bearish, 0 for neutral
    signals = [1 if p > last_price else -1 if p < last_price else 0 for p in preds]
    score = sum(signals)
    if score > 0:
        return "BULLISH 📈"
    elif score < 0:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

# ==========================================================
# Main Loop (Live)
# ==========================================================
try:
    while True:
        price = fetch_last_price(symbol)
        now = datetime.now(timezone.utc)
        price_history.append(price)
        if len(price_history) > window_size:
            price_history = price_history[-window_size:]

        # Predictions
        preds = [
            predict_lr(price_history),
            predict_hma(price_history),
            predict_kalman(price_history),
            predict_cwma(price_history),
            predict_dma(price_history)
        ]

        # Merge signals
        final_signal = merge_signals(preds, price)

        # Print output
        print(f"Signal: {final_signal}")

        time.sleep(interval)

except KeyboardInterrupt:
    print("Stopped live feed.")
