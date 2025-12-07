
!pip install python-binance pykalman websockets numpy --quiet

import websockets
import asyncio
import json
import numpy as np
from datetime import datetime, timezone
from collections import deque
from binance.client import Client
from pykalman import KalmanFilter

# ==========================================================
# PARAMETERS
# ==========================================================
symbol_rest = "BTCUSDT"
symbol_ws   = "btcusdt"
WS_URL = f"wss://stream.binance.us:9443/ws/{symbol_ws}@depth"

client = Client(tld="us", api_key="", api_secret="")

price_history = deque(maxlen=30)

# Orderbook memory
bids = {}
asks = {}
last_best_bid = None
last_best_ask = None

# For VPIN & volatility
vpin_window = deque(maxlen=50)
vol_window = deque(maxlen=100)



# ==========================================================
# TREND SIGNAL (From Code1)
# ==========================================================
def fetch_last_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])


def predict_lr(prices):
    if len(prices) < 2:
        return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m * len(prices) + c)


def predict_hma(prices, period=16):
    if len(prices) < period:
        return prices[-1]

    def wma(arr, n):
        if len(arr) < n:
            return arr[-1]
        w = np.arange(1, n+1)
        return np.sum(arr[-n:] * w) / np.sum(w)

    half = period // 2
    sqrtp = int(np.sqrt(period))

    h = wma(np.array(prices), half)
    f = wma(np.array(prices), period)
    raw = 2 * h - f

    return float(wma(np.array([raw]), sqrtp))


def predict_kalman(prices):
    if len(prices) < 2:
        return prices[-1]
    kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)
    state_means, _ = kf.smooth(np.array(prices))
    return float(state_means[-1])


def predict_cwma(prices):
    if len(prices) < 2:
        return prices[-1]
    returns = np.diff(prices)
    cov = np.cov(returns) if len(returns) > 1 else 1
    w = 1 / (1 + cov)
    return float(np.average(prices, weights=np.full(len(prices), w)))


def predict_dma(prices, disp=3):
    if len(prices) <= disp:
        return prices[-1]
    return float(np.mean(prices[-disp:]))


def merge_signals(preds, price):
    votes = [1 if p > price else -1 if p < price else 0 for p in preds]
    score = sum(votes)
    if score > 0:
        return "BULLISH 📈"
    elif score < 0:
        return "BEARISH 📉"
    return "NEUTRAL ➖"



# ==========================================================
# HFT INDICATORS (From Code2)
# ==========================================================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid] + asks[best_ask]
    return (best_bid * asks[best_ask] + best_ask * bids[best_bid]) / w


def spread_indicator():
    return min(asks.keys()) - max(bids.keys())


def order_flow_imbalance():
    global last_best_bid, last_best_ask
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    ofi = 0
    if last_best_bid is not None:
        ofi += best_bid - last_best_bid
    if last_best_ask is not None:
        ofi += last_best_ask - best_ask
    last_best_bid = best_bid
    last_best_ask = best_ask
    return ofi


def pressure_indicator(depth=5):
    tb = sorted(bids.keys(), reverse=True)[:depth]
    ta = sorted(asks.keys())[:depth]
    bp = sum(bids[p] for p in tb)
    ap = sum(asks[p] for p in ta)
    return bp, ap, (bp / ap if ap > 0 else None)


def orderbook_slope(depth=10):
    pr = sorted(list(bids.keys()) + list(asks.keys()))
    qu = [bids.get(p, asks.get(p, 0)) for p in pr]
    if len(pr) < 3:
        return 0
    return np.polyfit(pr, qu, 1)[0]


def inventory_imbalance(depth=5):
    tb = sorted(bids.keys(), reverse=True)[:depth]
    ta = sorted(asks.keys())[:depth]
    B = sum(bids[p] for p in tb)
    A = sum(asks[p] for p in ta)
    return (B-A)/(B+A+1e-9)


def vpin_indicator(midprice):
    vpin_window.append(midprice)
    if len(vpin_window) < vpin_window.maxlen:
        return None
    r = np.diff(vpin_window)
    b = np.sum(r > 0)
    s = np.sum(r < 0)
    return abs(b-s)/(b+s+1e-9)


def short_term_volatility(midprice):
    vol_window.append(midprice)
    if len(vol_window) < vol_window.maxlen:
        return None
    return np.std(np.diff(vol_window))


def liquidity_shock():
    spread = spread_indicator()
    if len(vol_window) < 10:
        return None
    return spread > 1.5*np.mean([abs(x) for x in vol_window])


def market_stress_index():
    spread = spread_indicator()
    vol = short_term_volatility(last_best_bid or 0)
    return spread * vol if vol else None



# ==========================================================
# MERGED ENGINE
# ==========================================================
async def depth_stream():
    print("🔵 Merged Trend Model + HFT Indicators\n")

    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # Update orderbook
            for price, qty in msg["b"]:
                p = float(price); q = float(qty)
                if q == 0: bids.pop(p, None)
                else: bids[p] = q

            for price, qty in msg["a"]:
                p = float(price); q = float(qty)
                if q == 0: asks.pop(p, None)
                else: asks[p] = q

            if not bids or not asks:
                continue

            # -----------------------------
            # REST Price for Trend Model
            # -----------------------------
            last_price = fetch_last_price(symbol_rest)
            price_history.append(last_price)

            preds = [
                predict_lr(list(price_history)),
                predict_hma(list(price_history)),
                predict_kalman(list(price_history)),
                predict_cwma(list(price_history)),
                predict_dma(list(price_history)),
            ]

            trend = merge_signals(preds, last_price)

            # -----------------------------
            # HFT Indicators
            # -----------------------------
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            mid = (best_bid + best_ask) / 2

            hft = {
                "microprice": microprice_indicator(),
                "ofi": order_flow_imbalance(),
                "spread": spread_indicator(),
                "bid_pressure": pressure_indicator()[0],
                "ask_pressure": pressure_indicator()[1],
                "pressure_ratio": pressure_indicator()[2],
                "orderbook_slope": orderbook_slope(),
                "imbalance": inventory_imbalance(),
                "vpin": vpin_indicator(mid),
                "volatility": short_term_volatility(mid),
                "liquidity_shock": liquidity_shock(),
                "stress": market_stress_index()
            }

            # -----------------------------
            # PRINT MERGED OUTPUT
            # -----------------------------
            print("\n⏱", datetime.utcnow(), "UTC")
            print("============================================")
            print("🔶 TREND SIGNAL (Merged ML Models):", trend)
            print("============================================")

            for k, v in hft.items():
                print(f"{k:18}: {v}")

            print("--------------------------------------------")


# Start
await depth_stream()
