
# =========================
# INSTALL DEPENDENCIES
# =========================
!pip install python-binance pykalman websockets nest_asyncio river --quiet

# =========================
# IMPORTS
# =========================
import asyncio, websockets, json
from collections import deque
import numpy as np
from datetime import datetime, timezone
from pykalman import KalmanFilter
from binance.client import Client
import nest_asyncio
from river import linear_model, preprocessing

nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
window_size = 30           # HFT midprice window
cache_window = 300         # River feature cache
price_history = []

# BinanceUS client
client = Client(tld="us", api_key="", api_secret="")

# WebSocket
ws_symbol = symbol.lower()
WS_URL = f"wss://stream.binance.us:9443/ws/{ws_symbol}@depth"

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
# UTILITY FUNCTIONS
# =========================
def trend_signal(pred, last_price):
    if pred > last_price: return "BULLISH 📈"
    elif pred < last_price: return "BEARISH 📉"
    return "NEUTRAL ➖"

# =========================
# TREND MODELS
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
    if not bids or not asks: return 0
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid]+asks[best_ask]
    return (best_bid*asks[best_ask]+best_ask*bids[best_bid])/w

def spread_indicator():
    return min(asks.keys()) - max(bids.keys()) if bids and asks else 0

def order_flow_imbalance():
    global last_best_bid,last_best_ask
    if not bids or not asks: return 0
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    ofi = 0
    if last_best_bid is not None: ofi += best_bid-last_best_bid
    if last_best_ask is not None: ofi += last_best_ask-best_ask
    last_best_bid,last_best_ask = best_bid,best_ask
    ofi_window.append(ofi)
    return ofi

def pressure_indicator(depth=5):
    if not bids or not asks: return 0,0,None
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0,0,None
    bid_pressure=sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_pressure=sum([asks[top_asks[i]] for i in range(available_levels)])
    ratio=bid_pressure/ask_pressure if ask_pressure>0 else None
    return bid_pressure,ask_pressure,ratio

# =========================
# FPGA-STYLE STREAMING FEATURES
# =========================
def weighted_imbalance(levels=5):
    if not bids or not asks: return 0
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(levels,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    imbalance=0
    weight_sum=0
    for i in range(available_levels):
        w=1/(i+1)
        b_qty=bids.get(top_bids[i],0)
        a_qty=asks.get(top_asks[i],0)
        imbalance+=w*(b_qty-a_qty)
        weight_sum+=w*(b_qty+a_qty)
    return imbalance/weight_sum if weight_sum!=0 else 0

def micro_momentum(price):
    micro_window.append(price)
    if len(micro_window)<2: return 0
    return micro_window[-1]-micro_window[0]

def price_skew(depth=5):
    if not bids or not asks: return 0
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    bid_vol=sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_vol=sum([asks[top_asks[i]] for i in range(available_levels)])
    return (bid_vol-ask_vol)/(bid_vol+ask_vol+1e-9)

# =========================
# RIVER ONLINE MODELS
# =========================
price_scaler = preprocessing.StandardScaler()
online_lr = linear_model.LinearRegression()
online_log = linear_model.LogisticRegression()

def update_river_models(midprice, features_dict):
    x = {**features_dict, "midprice": midprice}
    price_scaler.learn_one(x)
    x_scaled = price_scaler.transform_one(x)
    y_pred = online_lr.predict_one(x_scaled) or midprice
    online_lr.learn_one(x_scaled, midprice)
    trend = 1 if midprice > y_pred else -1 if midprice < y_pred else 0
    y_class = {1:"BULLISH 📈", -1:"BEARISH 📉", 0:"NEUTRAL ➖"}
    online_log.learn_one(x_scaled, trend)
    return y_pred, y_class[trend]

# =========================
# 1-MINUTE BAR PREDICTION
# =========================
def predict_next_bar(midprice, features_dict, max_dev_pct=0.001):
    """
    Predict next 1-min bar realistic target price
    - midprice: current midprice
    - features_dict: HFT features dict
    - max_dev_pct: maximum price variation relative to midprice (e.g., 0.1% -> 0.001)
    """
    # River prediction
    x = {**features_dict, "midprice": midprice}
    price_scaler.learn_one(x)
    x_scaled = price_scaler.transform_one(x)
    raw_pred = online_lr.predict_one(x_scaled) or midprice
    online_lr.learn_one(x_scaled, midprice)

    # Clamp prediction to small realistic deviation
    max_dev = midprice * max_dev_pct
    target_price = midprice + np.clip(raw_pred - midprice, -max_dev, max_dev)

    # Determine trend
    if target_price > midprice:
        trend = "BULLISH 📈"
    elif target_price < midprice:
        trend = "BEARISH 📉"
    else:
        trend = "NEUTRAL ➖"

    return target_price, trend


# =========================
# LIVE STREAM LOOP
# =========================
async def depth_stream():
    print("🔵 HFT + Predictive Engine (Python) - BinanceUS\n")
    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg=json.loads(msg)
            # Update orderbook
            for price, qty in msg.get("b", []):
                p,q=float(price),float(qty)
                if q==0: bids.pop(p,None)
                else: bids[p]=q
            for price, qty in msg.get("a", []):
                p,q=float(price),float(qty)
                if q==0: asks.pop(p,None)
                else: asks[p]=q
            if not bids or not asks: continue

            best_bid,max_best_ask=max(bids.keys()),min(asks.keys())
            midprice=(best_bid+max_best_ask)/2

            # Trend models
            price_history.append(midprice)
            if len(price_history)>window_size: price_history[:] = price_history[-window_size:]
            preds=[
                predict_lr(price_history),
                predict_hma(price_history),
                predict_kalman(price_history),
                predict_cwma(price_history),
                predict_dma(price_history)
            ]
            trend_final = merge_signals(preds, midprice)
            signals_dict = { "LR": trend_signal(preds[0], midprice),
                             "HMA": trend_signal(preds[1], midprice),
                             "Kalman": trend_signal(preds[2], midprice),
                             "CWMA": trend_signal(preds[3], midprice),
                             "DMA": trend_signal(preds[4], midprice) }

            # HFT Features
            ofi = order_flow_imbalance()
            bid_p, ask_p, ratio = pressure_indicator()
            hft_features = {
                "microprice": microprice_indicator(),
                "ofi": ofi,
                "spread": spread_indicator(),
                "bid_pressure": bid_p,
                "ask_pressure": ask_p,
                "pressure_ratio": ratio,
                "imbalance": weighted_imbalance(),
                "micro_momentum": micro_momentum(midprice),
                "price_skew": price_skew()
            }

            # River + next 1-min bar prediction
            next_price, next_trend = predict_next_bar(midprice, hft_features)

            # Print outputs
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ Trend Models:", trend_final)
            for k,v in signals_dict.items(): print(f"   {k:7}: {v}")
            print("⭐ HFT Indicators:")
            for k,v in hft_features.items(): print(f"   {k:18}: {v}")
            print("⭐ Next 1-min Prediction:", f"{next_trend} | Target Price: {next_price:.2f}")
            print("------------------------------------------------------------")

# =========================
# RUN
# =========================
await depth_stream()
