
# =========================
# INSTALL DEPENDENCIES
# =========================
!pip install python-binance pykalman websockets nest_asyncio --quiet

# =========================
# IMPORTS
# =========================
import asyncio
import websockets
import json
from collections import deque
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import time
from pykalman import KalmanFilter
from binance.client import Client
import nest_asyncio
nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
interval = 1  # seconds for price fetch
window_size = 30  # lookback for trend models
cache_window = 300  # last 5 minutes for ML prediction
price_history = []

# Binance Client
client = Client(tld="us", api_key="", api_secret="")

# WebSocket
ws_symbol = symbol.lower()
WS_URL = f"wss://stream.binance.us:9443/ws/{ws_symbol}@depth"

# Orderbook Memory
bids = {}
asks = {}
last_best_bid = None
last_best_ask = None
vpin_window = deque(maxlen=50)
vol_window = deque(maxlen=100)
ofi_window = deque(maxlen=20)
micro_window = deque(maxlen=10)
cancel_window = deque(maxlen=50)
snapshot_cache = deque(maxlen=cache_window)

# =========================
# UTILITY FUNCTIONS
# =========================
def trend_signal(pred, last_price):
    if pred > last_price:
        return "BULLISH 📈"
    elif pred < last_price:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

def fetch_last_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

# =========================
# TREND PREDICTION MODELS
# =========================
def predict_lr(prices):
    if len(prices) < 2: return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m*len(prices)+c)

def predict_hma(prices, period=16):
    if len(prices) < period: return prices[-1]
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

def merge_signals(preds, last_price):
    signals = [1 if p>last_price else -1 if p<last_price else 0 for p in preds]
    score = sum(signals)
    if score>0: return "BULLISH 📈"
    elif score<0: return "BEARISH 📉"
    return "NEUTRAL ➖"

# =========================
# HFT INDICATORS
# =========================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid] + asks[best_ask]
    return (best_bid*asks[best_ask]+best_ask*bids[best_bid])/w

def spread_indicator():
    return min(asks.keys()) - max(bids.keys())

def order_flow_imbalance():
    global last_best_bid, last_best_ask
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    ofi = 0
    if last_best_bid is not None: ofi += best_bid - last_best_bid
    if last_best_ask is not None: ofi += last_best_ask - best_ask
    last_best_bid, last_best_ask = best_bid, best_ask
    ofi_window.append(ofi)
    return ofi

def pressure_indicator(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels==0: return 0,0,None
    bid_pressure = sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_pressure = sum([asks[top_asks[i]] for i in range(available_levels)])
    ratio = bid_pressure/ask_pressure if ask_pressure>0 else None
    return bid_pressure, ask_pressure, ratio

def orderbook_slope(depth=10):
    prices = sorted(list(bids.keys()) + list(asks.keys()))
    quantities = [bids.get(p, asks.get(p,0)) for p in prices]
    if len(prices)<3: return 0
    return np.polyfit(prices, quantities,1)[0]

def inventory_imbalance(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels==0: return 0
    B = sum([bids[top_bids[i]] for i in range(available_levels)])
    A = sum([asks[top_asks[i]] for i in range(available_levels)])
    return (B-A)/(B+A+1e-9)

def vpin_indicator(price):
    vpin_window.append(price)
    if len(vpin_window)<vpin_window.maxlen: return None
    returns = np.diff(vpin_window)
    buy_volume = np.sum(returns>0)
    sell_volume = np.sum(returns<0)
    return abs(buy_volume-sell_volume)/(buy_volume+sell_volume+1e-9)

def short_term_volatility(price):
    vol_window.append(price)
    if len(vol_window)<vol_window.maxlen: return None
    return np.std(np.diff(vol_window))

def liquidity_shock():
    spread = spread_indicator()
    return spread > 1.5*np.mean([abs(x) for x in vol_window]) if len(vol_window)>10 else None

# =========================
# FPGA-STYLE HFT FEATURES (SAFE)
# =========================
def weighted_imbalance(levels=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(levels, len(top_bids), len(top_asks))
    if available_levels==0: return 0
    imbalance = 0
    weight_sum = 0
    for i in range(available_levels):
        w = 1/(i+1)
        b_qty = bids.get(top_bids[i],0)
        a_qty = asks.get(top_asks[i],0)
        imbalance += w*(b_qty - a_qty)
        weight_sum += w*(b_qty + a_qty)
    return imbalance/weight_sum if weight_sum!=0 else 0

def rolling_ofi_sum():
    return sum(ofi_window)

def micro_momentum(price):
    micro_window.append(price)
    if len(micro_window)<2: return 0
    return micro_window[-1]-micro_window[0]

def cancellation_ratio(msg):
    cancels = sum(1 for p,q in msg.get("b",[]) if q==0) + sum(1 for p,q in msg.get("a",[]) if q==0)
    cancel_window.append(cancels)
    return np.mean(cancel_window)

def price_skew(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels==0: return 0
    bid_vol = sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_vol = sum([asks[top_asks[i]] for i in range(available_levels)])
    return (bid_vol - ask_vol)/(bid_vol + ask_vol + 1e-9)

# =========================
# LIVE STREAM PROCESSING
# =========================
async def depth_stream():
    print("🔵 Merged Trend + HFT + FPGA-style Models (SAFE)\n")

    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # Update orderbook
            for price, qty in msg["b"]:
                p=float(price); q=float(qty)
                if q==0: bids.pop(p,None)
                else: bids[p]=q
            for price, qty in msg["a"]:
                p=float(price); q=float(qty)
                if q==0: asks.pop(p,None)
                else: asks[p]=q

            if not bids or not asks: continue

            # Latest midprice
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            midprice = (best_bid+best_ask)/2

            # ------------------------
            # Trend Models
            # ------------------------
            price_history.append(midprice)
            if len(price_history)>window_size: price_history[:] = price_history[-window_size:]
            preds = [
                predict_lr(price_history),
                predict_hma(price_history),
                predict_kalman(price_history),
                predict_cwma(price_history),
                predict_dma(price_history)
            ]
            trend_signal_final = merge_signals(preds, midprice)

            # Individual trend signals
            signals_dict = {
                "LR": trend_signal(preds[0], midprice),
                "HMA": trend_signal(preds[1], midprice),
                "Kalman": trend_signal(preds[2], midprice),
                "CWMA": trend_signal(preds[3], midprice),
                "DMA": trend_signal(preds[4], midprice)
            }

            # ------------------------
            # HFT Indicators
            # ------------------------
            ofi = order_flow_imbalance()
            bid_p, ask_p, ratio = pressure_indicator()
            hft_features = {
                "microprice": microprice_indicator(),
                "ofi": ofi,
                "spread": spread_indicator(),
                "bid_pressure": bid_p,
                "ask_pressure": ask_p,
                "pressure_ratio": ratio,
                "orderbook_slope": orderbook_slope(),
                "imbalance": inventory_imbalance(),
                "vpin": vpin_indicator(midprice),
                "volatility": short_term_volatility(midprice),
                "liquidity_shock": liquidity_shock()
            }

            # ------------------------
            # FPGA-style features
            # ------------------------
            w_imb = weighted_imbalance()
            r_ofi = rolling_ofi_sum()
            micro_mom = micro_momentum(midprice)
            cancel_r = cancellation_ratio(msg)
            p_skew = price_skew()
            fpga_features = {
                "weighted_imbalance": (w_imb, trend_signal(w_imb, 0)),
                "rolling_ofi": (r_ofi, trend_signal(r_ofi,0)),
                "micro_momentum": (micro_mom, trend_signal(micro_mom,0)),
                "cancel_ratio": (cancel_r, trend_signal(cancel_r,0)),
                "price_skew": (p_skew, trend_signal(p_skew,0))
            }

            # ------------------------
            # Cache for 5-min prediction
            # ------------------------
            snapshot_cache.append({
                "midprice": midprice,
                **hft_features,
                **{k:v[0] for k,v in fpga_features.items()}
            })

            # ------------------------
            # PRINT CLEAN OUTPUT
            # ------------------------
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ Trend Models:", trend_signal_final)
            for k,v in signals_dict.items():
                print(f"   {k:7}: {v}")
            print("⭐ HFT Indicators:")
            for k,v in hft_features.items():
                print(f"   {k:18}: {v}")
            print("⭐ FPGA-style HFT Features:")
            for k,(val,sig) in fpga_features.items():
                print(f"   {k:18}: {val} | {sig}")
            print("------------------------------------------------------------")

# =========================
# RUN
# =========================
await depth_stream()
