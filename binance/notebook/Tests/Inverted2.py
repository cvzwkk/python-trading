

# =========================
# INSTALL DEPENDENCIES
# =========================
#!pip install python-binance pykalman websockets nest_asyncio river scipy --quiet

# =========================
# IMPORTS
# =========================
import requests
import asyncio
import websockets
import json
from collections import deque
import numpy as np
from datetime import datetime
from pykalman import KalmanFilter
from binance.client import Client
import nest_asyncio
from river import linear_model, preprocessing
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers, models

nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
interval = 1  # seconds for price fetch
window_size = 30  # trend model window
cache_window = 300  # last 5 minutes for River ML
price_history = []
SYMBOL = "BTCUSDT"
ROWS = 90  # top rows to sum
# Binance client
client = Client(tld="us", api_key="", api_secret="")

# WebSocket
ws_symbol = symbol.lower()
WS_URL = f"wss://stream.binance.us:9443/ws/{ws_symbol}@depth"

# Correct symbol mapping per exchange
EXCHANGE_SYMBOL = {
    "binance": lambda s: s,          # BTCUSDT
    "kraken": lambda s: "XBTUSDT",   # Kraken uses XBT
    "kucoin": lambda s: s.replace("USDT","-USDT"),
    "huobi": lambda s: s.lower(),
    "bybit": lambda s: s,            # BTCUSDT
    "okx": lambda s: s.replace("USDT","-USDT")
}

def safe_json_get(url):
    try:
        r = requests.get(url, timeout=5)
        r.raise_for_status()
        return r.json()
    except:
        return None

# ====== Exchange functions ======
def get_binance(symbol):
    r = safe_json_get(f"https://api.binance.com/api/v3/depth?symbol={symbol}&limit={ROWS}")
    if r:
        return r.get("bids", []), r.get("asks", [])
    return [], []

def get_kraken(symbol):
    r = safe_json_get(f"https://api.kraken.com/0/public/Depth?pair={symbol}&count={ROWS}")
    if r and "result" in r:
        key = list(r["result"].keys())[0]
        return r["result"][key]["bids"], r["result"][key]["asks"]
    return [], []

def get_kucoin(symbol):
    r = safe_json_get(f"https://api.kucoin.com/api/v1/market/orderbook/level2_100?symbol={symbol}")
    if r and "data" in r:
        return r["data"]["bids"], r["data"]["asks"]
    return [], []

def get_huobi(symbol):
    r = safe_json_get(f"https://api.huobi.pro/market/depth?symbol={symbol}&type=step0")
    if r and "tick" in r:
        return r["tick"]["bids"][:ROWS], r["tick"]["asks"][:ROWS]
    return [], []

def get_bybit(symbol):
    r = safe_json_get(f"https://api.bybit.com/v5/market/books?instId={symbol}&sz={ROWS}")
    if r and "data" in r and len(r["data"]) > 0:
        data = r["data"][0]
        return data["bids"], data["asks"]
    return [], []

def get_okx(symbol):
    r = safe_json_get(f"https://www.okx.com/api/v5/market/books?instId={symbol}&sz={ROWS}")
    if r and "data" in r and len(r["data"]) > 0:
        data = r["data"][0]
        return data["bids"], data["asks"]
    return [], []

EXCHANGES = {
    "Binance": get_binance,
    "Kraken": get_kraken,
    "Kucoin": get_kucoin,
    "Huobi": get_huobi,
    "Bybit": get_bybit,
    "OKX": get_okx
}

# ====== Aggregate bids and asks ======
total_bid_amount = 0.0
total_ask_amount = 0.0

for name, func in EXCHANGES.items():
    try:
        mapped_symbol = EXCHANGE_SYMBOL[name.lower()](SYMBOL)
        bids, asks = func(mapped_symbol)
        if not bids:
            print(f"{name} returned empty bids.")
        if not asks:
            print(f"{name} returned empty asks.")
        total_bid_amount += sum(float(b[1]) for b in bids[:ROWS])
        total_ask_amount += sum(float(a[1]) for a in asks[:ROWS])
    except Exception as e:
        print(f"{name} error: {e}")



# Orderbook
bids, asks = {}, {}
last_best_bid, last_best_ask = None, None
vpin_window, vol_window, ofi_window, micro_window, cancel_window = (
    deque(maxlen=50),
    deque(maxlen=100),
    deque(maxlen=20),
    deque(maxlen=10),
    deque(maxlen=50)
)
snapshot_cache = deque(maxlen=cache_window)

# =========================
# MULTI-TRADING SYSTEM
# =========================
balance = 1000.0
positions = {}  # key: strategy_name, value: {"type": "LONG"/"SHORT", "entry": price}
trade_log = []

# =========================
# UTILITY FUNCTIONS
# =========================
def invert_signal(sig):
    if isinstance(sig, str):
        if "BUY" in sig: return "SELL 🔴"
        if "SELL" in sig: return "BUY 🟢"
    return sig

def trend_signal(pred, last_price):
    if pred > last_price: return "BUY 🟢"
    elif pred < last_price: return "SELL 🔴"
    return "NEUTRAL ➖"

def fetch_last_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def merge_signals(preds, last_price):
    signals = [1 if p>last_price else -1 if p<last_price else 0 for p in preds]
    score = sum(signals)
    if score>0: return "BUY 🟢"
    elif score<0: return "SELL 🔴"
    return "NEUTRAL ➖"

# =========================
# TREND PREDICTION MODELS
# =========================
def predict_lr(prices):
    if len(prices)<2: return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    m,c = np.linalg.lstsq(A,y,rcond=None)[0]
    return float(m*len(prices)+c)

def predict_hma(prices, period=16):
    if len(prices)<period: return prices[-1]
    def wma(arr,n):
        if len(arr)<n: return arr[-1]
        weights = np.arange(1,n+1)
        return np.sum(arr[-n:]*weights)/weights.sum()
    half = period//2
    sqrt_len = int(np.sqrt(period))
    wma_half = wma(np.array(prices), half)
    wma_full = wma(np.array(prices), period)
    raw_hma = 2*wma_half - wma_full
    return float(wma(np.array([raw_hma]), sqrt_len))

def predict_kalman(prices):
    if len(prices)<2: return prices[-1]
    kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)
    state_means,_ = kf.smooth(np.array(prices))
    return float(state_means[-1])

def predict_cwma(prices):
    if len(prices)<2: return prices[-1]
    returns = np.diff(prices)
    cov = np.cov(returns) if len(returns)>1 else 1.0
    weight = 1/(1+cov)
    return float(np.average(prices, weights=np.full(len(prices), weight)))

def predict_dma(prices, displacement=3):
    if len(prices)<=displacement: return prices[-1]
    return np.mean(prices[-displacement:])

def predict_ema(prices, period=10):
    if len(prices)<period: return prices[-1]
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode='valid')[0])

def predict_tema(prices, period=10):
    if len(prices)<period*3: return prices[-1]
    ema1 = predict_ema(prices, period)
    ema2 = predict_ema([predict_ema(prices[:i+1], period) for i in range(len(prices))], period)
    ema3 = predict_ema([predict_ema([predict_ema(prices[:i+1], period) for i in range(j+1)], period) for j in range(len(prices))], period)
    return float(3*ema1 - 3*ema2 + ema3)

def predict_wma(prices, period=10):
    if len(prices) < period: return prices[-1]
    weights = np.arange(1, period+1)
    return float(np.dot(prices[-period:], weights)/weights.sum())

def predict_smma(prices, period=10):
    if len(prices) < period: return prices[-1]
    smma = np.mean(prices[:period])
    for p in prices[period:]:
        smma = (smma*(period-1)+p)/period
    return float(smma)

def predict_momentum(prices, period=5):
    if len(prices) < period+1: return prices[-1]
    return prices[-1] - prices[-period-1]

# =========================
# EXOTIC MODELS
# =========================
def predict_hzlog(prices, period=14):
    if len(prices)<period: return prices[-1]
    log_prices = np.log(prices[-period:])
    analytic_signal = scipy.signal.hilbert(log_prices)
    hz_signal = np.real(analytic_signal[-1])
    return float(np.exp(hz_signal))

def predict_vydia(prices, period=10):
    if len(prices)<period: return prices[-1]
    vol = np.std(prices[-period:])
    weights = np.exp(np.linspace(-vol, 0., period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode='valid')[0])

def predict_parma(prices, period=14):
    if len(prices)<period: return prices[-1]
    highs = np.array(prices[-period:])
    lows = np.array(prices[-period:])
    weights = highs - lows + 1e-9
    return float(np.average(prices[-period:], weights=weights))

def predict_junx(prices, period=5):
    if len(prices)<period+1: return prices[-1]
    diffs = np.diff(prices[-period-1:])
    jump_adj = np.sum(diffs[diffs>0]) - np.sum(diffs[diffs<0])
    return float(prices[-1] + jump_adj/period)

def predict_t3(prices, period=10, vfactor=0.7):
    if len(prices)<period*3: return prices[-1]
    def ema(arr, p): return float(np.convolve(arr[-p:], np.exp(np.linspace(-1.,0.,p))/np.exp(np.linspace(-1.,0.,p)).sum(), mode='valid')[0])
    e1 = ema(prices, period)
    e2 = ema([ema(prices[:i+1], period) for i in range(len(prices))], period)
    e3 = ema([ema([ema(prices[:i+1], period) for i in range(j+1)], period) for j in range(len(prices))], period)
    return float(e1*(1+vfactor) - e2*vfactor + e3*(vfactor**2))

def predict_ichimoku(prices, short=9, long=26):
    if len(prices)<long: return prices[-1]
    tenkan = (max(prices[-short:]) + min(prices[-short:]))/2
    kijun  = (max(prices[-long:]) + min(prices[-long:]))/2
    cloud_top = max(tenkan,kijun)
    cloud_bot = min(tenkan,kijun)
    if prices[-1]>cloud_top: return prices[-1]*1.001
    elif prices[-1]<cloud_bot: return prices[-1]*0.999
    else: return prices[-1]

# =========================
# ADDITIONAL EXOTIC MODELS
# =========================

def predict_ar(prices, lags=8):
    """Autoregressive-like predictor using ordinary least squares on lagged values."""
    if len(prices) <= lags:
        return prices[-1]
    y = np.array(prices[lags:])
    X = np.column_stack([np.array(prices[lags - i: -i]) for i in range(1, lags + 1)])
    # add intercept
    A = np.vstack([X.T, np.ones(X.shape[0])]).T
    try:
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        last_row = np.array(prices[-lags:])[::-1]  # most recent lags (in same order used)
        pred = last_row.dot(coef[:-1]) + coef[-1]
        return float(pred)
    except Exception:
        return float(prices[-1])

def predict_fft(prices, keep=6, horizon=1):
    """FFT-based seasonal extrapolation: keep top `keep` spectral components and extrapolate."""
    n = len(prices)
    if n < 6:
        return prices[-1]
    x = np.array(prices) - np.mean(prices)
    freqs = np.fft.rfft(x)
    mags = np.abs(freqs)
    # zero out small components
    idx = np.argsort(mags)[-keep:]
    mask = np.zeros_like(freqs, dtype=bool)
    mask[idx] = True
    filtered = freqs * mask
    # reconstruct and linearly extrapolate the next step by assuming same phase progression
    recon = np.fft.irfft(filtered, n=n)
    # fall back to last reconstructed value + slope from last two reconstructed
    if len(recon) >= 2:
        slope = recon[-1] - recon[-2]
        return float((recon[-1] + slope * horizon) + np.mean(prices))
    else:
        return float(prices[-1])

def predict_macd_midprice(prices, fast=12, slow=26, signal=9):
    """Return a projection based on MACD momentum added to last price (small-step projection)."""
    if len(prices) < slow:
        return prices[-1]
    def ema(arr, n):
        weights = np.exp(np.linspace(-1., 0., n))
        weights /= weights.sum()
        return np.convolve(arr, weights, mode='valid')
    fast_series = ema(prices, fast)
    slow_series = ema(prices, slow)
    if len(fast_series) < 1 or len(slow_series) < 1:
        return prices[-1]
    # align ends
    macd = fast_series[-len(slow_series):] - slow_series
    if len(macd) < signal:
        macd_signal = np.mean(macd)
    else:
        macd_signal = ema(macd, signal)[-1]
    macd_hist = macd[-1] - macd_signal
    # project a small step proportional to macd_hist
    return float(prices[-1] + 0.5 * macd_hist)

def predict_median_filter(prices, period=7):
    """Robust median-filter prediction (resistant to spikes)."""
    if len(prices) < period:
        return prices[-1]
    return float(np.median(prices[-period:]))

def predict_entropy_weighted(prices, period=20):
    """Weights recent prices by inverse normalised return entropy (lower entropy -> higher weight)."""
    if len(prices) < 6:
        return prices[-1]
    window = np.array(prices[-period:])
    ret = np.diff(window)
    # small probabilistic entropy proxy: use normalized absolute returns distribution
    if len(ret) < 2:
        return prices[-1]
    probs, _ = np.histogram(np.abs(ret), bins=6, density=True)
    probs = probs + 1e-9
    probs /= probs.sum()
    entropy = -np.sum(probs * np.log(probs))
    # produce a weight inversely proportional to entropy; apply to recent values
    win_len = min(period, len(window))
    weights = 1.0 / (1.0 + np.linspace(entropy, entropy * 0.5, win_len))
    weights /= weights.sum()
    return float(np.dot(window[-win_len:], weights))

def predict_rls_trend(prices):
    """Simple batch Recursive Least Squares-like trend projection (forgetting via exponential window)."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    lam = 0.97  # forgetting factor
    # fit line with exponential weights
    x = np.arange(n)
    w = lam ** (n - 1 - x)  # recent points higher weight
    W = np.diag(w)
    A = np.vstack([x, np.ones_like(x)]).T
    try:
        beta = np.linalg.lstsq(A.T @ W @ A, A.T @ W @ np.array(prices), rcond=None)[0]
        next_x = n
        return float(beta[0] * next_x + beta[1])
    except Exception:
        return float(prices[-1])

# =========================
# MORE EXOTIC MODELS (paste under your ADDITIONAL EXOTIC MODELS)
# =========================

import math
from scipy.signal import savgol_filter

def predict_savgol(prices, window=9, polyorder=3, horizon=1):
    """S-G denoise + last-slope extrapolation. Fast and robust to spikes."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    # ensure odd window and <= n
    w = min(window, n if n % 2 == 1 else n-1)
    if w < 3:
        return prices[-1]
    try:
        filt = savgol_filter(prices[-w:], w, min(polyorder, w-1))
        slope = filt[-1] - filt[-2]
        return float(filt[-1] + slope * horizon)
    except Exception:
        return float(prices[-1])

def predict_local_poly(prices, window=12, degree=2, horizon=1):
    """Fit low-degree polynomial to last `window` points and extrapolate."""
    n = len(prices)
    if n < degree + 2:
        return prices[-1]
    w = min(window, n)
    y = np.array(prices[-w:])
    x = np.arange(w).astype(float)
    # shift x so poly is numerically stable (predict next index = w)
    x_shift = x - x.mean()
    coeffs = np.polyfit(x_shift, y, min(degree, w-1))
    poly = np.poly1d(coeffs)
    next_x = (w) - x.mean()
    return float(poly(next_x))

def predict_holt(prices, alpha=0.4, beta=0.2, horizon=1):
    """Simple Holt's linear trend (double exp smoothing) 1-step forecast."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    s = prices[0]
    b = prices[1] - prices[0]
    for i in range(1, n):
        last = s
        s = alpha * prices[i] + (1 - alpha) * (s + b)
        b = beta * (s - last) + (1 - beta) * b
    return float(s + b * horizon)

def predict_lms_online(prices, taps=6, mu=0.01):
    """Simple batch LMS/SGD linear predictor using last `taps` values.
       Very cheap and adaptive: fits weights to minimize last-window squared error then forecasts 1-step."""
    n = len(prices)
    if n <= taps:
        return prices[-1]
    # prepare X matrix of lagged values and y vector
    X = []
    y = []
    for i in range(taps, n):
        X.append(prices[i-taps:i][::-1])  # most recent first
        y.append(prices[i])
    X = np.array(X)
    y = np.array(y)
    # initialize weights as zeros and do a few LMS passes (cheap)
    w = np.zeros(taps)
    for xi, yi in zip(X, y):
        pred = w.dot(xi)
        e = yi - pred
        w = w + mu * e * xi
    last_row = np.array(prices[-taps:][::-1])
    return float(w.dot(last_row))

def predict_burg_ar(prices, order=6):
    """Burg algorithm to estimate AR coefficients then 1-step forecast.
       Order should be small for 1s timeframe (3..8)."""
    x = np.array(prices)
    n = len(x)
    if n <= 2 or order < 1:
        return float(prices[-1])
    m = min(order, n-1)
    # initialize
    ef = x.copy()
    eb = x.copy()
    a = np.zeros(m+1)
    a[0] = 1.0
    den = np.dot(ef, ef)
    # reflection coefficients
    coeffs = np.zeros(m)
    for k in range(m):
        # compute numerator and denominator
        num = -2.0 * np.dot(ef[k+1:], eb[k:n-1])
        den = np.dot(ef[k+1:], ef[k+1:]) + np.dot(eb[k:n-1], eb[k:n-1])
        if abs(den) < 1e-12:
            break
        gamma = num / den
        coeffs[k] = gamma
        # update forward/backward errors
        ef_next = ef.copy()
        eb_next = eb.copy()
        ef_next[k+1:n] = ef[k+1:n] + gamma * eb[k:n-1]
        eb_next[k+1:n] = eb[k+1:n] + gamma * ef[k+1:n]
        ef = ef_next
        eb = eb_next
    # convert reflection coefficients to AR coefs (Levinson-Durbin style)
    ar = np.array([0.0]*m)
    for i in range(m):
        k = coeffs[i]
        ar_prev = ar[:i].copy()
        ar[:i] = ar_prev - k * ar_prev[::-1]
        ar[i] = k
    # forecast using last m samples: x_hat = -sum(ar * x_{t - i - 1})
    last_vals = x[-m:][::-1] if m>0 else np.array([x[-1]])
    forecast = -np.dot(ar, last_vals) + x[-1]  # add last to reduce bias
    return float(forecast)

def predict_median_trend(prices, window=10, horizon=1):
    """Take median of local slopes inside window and extrapolate."""
    n = len(prices)
    if n < 3:
        return prices[-1]
    w = min(window, n-1)
    slopes = []
    arr = np.array(prices[-(w+1):])
    for i in range(len(arr)-1):
        slopes.append(arr[i+1] - arr[i])
    med_slope = float(np.median(slopes))
    return float(prices[-1] + med_slope * horizon)

def predict_envelope(prices, window=12):
    """Use local min/max envelope bias to create a tiny projected movement.
       Good when price is trapped in micro-range and you want envelope push."""
    n = len(prices)
    if n < 6:
        return prices[-1]
    w = min(window, n)
    seg = np.array(prices[-w:])
    local_max = np.max(seg)
    local_min = np.min(seg)
    center = (local_max + local_min) / 2.0
    bias = (seg[-1] - center) / (local_max - local_min + 1e-9)
    # bias in [-1,1] scaled to a small step
    step = (local_max - local_min) * 0.15
    return float(seg[-1] + bias * step)

# ========================
# TensorFlow Models
# ========================
def predict_tf_lstm(prices, window=20):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    model = models.Sequential([
        layers.LSTM(16, return_sequences=False),
        layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    # quick training (only 1 epoch for speed)
    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1, window, 1)
    y = np.array(prices[window:])
    if len(X) < 2:
        return prices[-1]

    model.fit(X, y, batch_size=8, epochs=1, verbose=0)

    pred = model.predict(seq, verbose=0)[0][0]
    return float(pred)

def predict_tf_cnn(prices, window=32):
    if len(prices) < window:
        return prices[-1]

    seq = np.array(prices[-window:], dtype=np.float32).reshape(1, window, 1)

    model = models.Sequential([
        layers.Conv1D(32, 3, activation='relu'),
        layers.Conv1D(16, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    # Prepare training data
    X = np.array([prices[i-window:i] for i in range(window, len(prices))]).reshape(-1, window, 1)
 
