
# ============================
# Full merged script
# ============================
# Installs (Colab)
!pip install python-binance pykalman websockets scikit-learn --quiet

import asyncio
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from collections import deque, defaultdict
import websockets
from binance.client import Client
from pykalman import KalmanFilter
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------
# Params
# ------------------------------------------------------------
# REST symbol (for last price)
SYMBOL_REST = "BTCUSDT"
# WS symbol (lowercase)
SYMBOL_WS = "btcusdt"
WS_URL = f"wss://stream.binance.us:9443/ws/{SYMBOL_WS}@depth"

# Rolling windows & caches
PRICE_HISTORY_LEN = 300             # used by short-term predictions (seconds)
CACHE_SECONDS = 1800                # how many seconds of snapshots to keep (~30 minutes)
SNAPSHOT_MAX = 2000                 # safety cap on number of snapshots
snapshot_cache = deque(maxlen=SNAPSHOT_MAX)   # each entry = dict with features + timestamp

# Price history for classic predictions
price_history = deque(maxlen=PRICE_HISTORY_LEN)

# Orderbook memory (silent orderbook)
bids = {}    # price -> qty
asks = {}    # price -> qty
last_best_bid = None
last_best_ask = None

# VPIN & vol windows
vpin_window = deque(maxlen=50)
vol_window = deque(maxlen=100)

# Binance REST public client (no API keys required for ticker)
client = Client(tld="us", api_key="", api_secret="")

# -------------------------------
# Prediction functions (from code1)
# -------------------------------
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
        weights = np.arange(1, n+1)
        return np.sum(arr[-n:] * weights) / weights.sum()
    half = period // 2
    sqrtp = int(np.sqrt(period))
    wma_half = wma(np.array(prices), half)
    wma_full = wma(np.array(prices), period)
    raw_hma = 2 * wma_half - wma_full
    final_hma = wma(np.array([raw_hma]), sqrtp)
    return float(final_hma)

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
    cov = np.cov(returns) if len(returns) > 1 else 1.0
    weight = 1 / (1 + cov)
    weighted_avg = np.average(prices, weights=np.full(len(prices), weight))
    return float(weighted_avg)

def predict_dma(prices, displacement=3):
    if len(prices) <= displacement:
        return prices[-1]
    return float(np.mean(prices[-displacement:]))

def merge_signals(preds, last_price):
    signals = [1 if p > last_price else -1 if p < last_price else 0 for p in preds]
    score = sum(signals)
    if score > 0:
        return "BULLISH 📈"
    elif score < 0:
        return "BEARISH 📉"
    return "NEUTRAL ➖"

# -------------------------------
# HFT Indicators (from code2)
# -------------------------------
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
    top_bids = sorted(bids.keys(), reverse=True)[:depth]
    top_asks = sorted(asks.keys())[:depth]
    bid_pressure = sum(bids[p] for p in top_bids)
    ask_pressure = sum(asks[p] for p in top_asks)
    ratio = bid_pressure / ask_pressure if ask_pressure > 0 else None
    return bid_pressure, ask_pressure, ratio

def orderbook_slope(depth=10):
    prices = sorted(list(bids.keys()) + list(asks.keys()))
    quantities = [bids.get(p, asks.get(p, 0)) for p in prices]
    if len(prices) < 3:
        return 0.0
    return float(np.polyfit(prices, quantities, 1)[0])

def inventory_imbalance(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)[:depth]
    top_asks = sorted(asks.keys())[:depth]
    B = sum(bids[p] for p in top_bids)
    A = sum(asks[p] for p in top_asks)
    return (B - A) / (B + A + 1e-9)

def vpin_indicator(price):
    vpin_window.append(price)
    if len(vpin_window) < vpin_window.maxlen:
        return None
    returns = np.diff(vpin_window)
    buy_volume = np.sum(returns > 0)
    sell_volume = np.sum(returns < 0)
    return abs(buy_volume - sell_volume) / (buy_volume + sell_volume + 1e-9)

def short_term_volatility(price):
    vol_window.append(price)
    if len(vol_window) < vol_window.maxlen:
        return None
    return float(np.std(np.diff(vol_window)))

def liquidity_shock():
    spread = spread_indicator()
    if len(vol_window) < 10:
        return None
    return spread > 1.5 * np.mean([abs(x) for x in vol_window])

def market_stress_index():
    try:
        spread = spread_indicator()
        vol = short_term_volatility(last_best_bid or 0)
        if vol is None:
            return None
        return spread * vol
    except Exception:
        return None

# -------------------------------
# Helper: fetch last price via REST
# -------------------------------
def fetch_last_price(symbol=SYMBOL_REST):
    try:
        ticker = client.get_symbol_ticker(symbol=symbol)
        return float(ticker['price'])
    except Exception:
        # fallback: average best bid/ask if available
        if bids and asks:
            return (max(bids.keys()) + min(asks.keys())) / 2.0
        return None

# -------------------------------
# Snapshot / Cache logic for 5-min prediction
# -------------------------------
def make_snapshot(mid, micro, ofi, spread, bid_pressure, ask_pressure, pressure_ratio, slope, imbalance, vpin, volatility):
    ts = datetime.utcnow().timestamp()
    return {
        "timestamp": ts,
        "mid": float(mid),
        "micro": float(micro) if micro is not None else float(mid),
        "ofi": float(ofi) if ofi is not None else 0.0,
        "spread": float(spread) if spread is not None else 0.0,
        "bid_pressure": float(bid_pressure) if bid_pressure is not None else 0.0,
        "ask_pressure": float(ask_pressure) if ask_pressure is not None else 0.0,
        "pressure_ratio": float(pressure_ratio) if pressure_ratio is not None else 0.0,
        "slope": float(slope) if slope is not None else 0.0,
        "imbalance": float(imbalance) if imbalance is not None else 0.0,
        "vpin": float(vpin) if vpin is not None else 0.0,
        "volatility": float(volatility) if volatility is not None else 0.0
    }

def add_snapshot(snapshot):
    # Keep cache within time window (CACHE_SECONDS)
    snapshot_cache.append(snapshot)
    # Optionally prune by timestamp to keep only last CACHE_SECONDS
    cutoff = datetime.utcnow().timestamp() - CACHE_SECONDS
    while len(snapshot_cache) and snapshot_cache[0]["timestamp"] < cutoff:
        snapshot_cache.popleft()

# -------------------------------
# Online model: SGDClassifier for 5-minute up/down
# -------------------------------
FEATURE_COLS = [
    "mid", "micro", "ofi", "spread", "bid_pressure", "ask_pressure", "pressure_ratio",
    "slope", "imbalance", "vpin", "volatility",
    # engineered features (will be computed from df)
    "r1", "r3", "r5",
    "d_micro", "d_ofi", "d_spread", "d_slope", "d_vpin"
]

scaler = StandardScaler()
clf = SGDClassifier(loss="log", max_iter=1000, tol=1e-3)
clf_initialized = False
MIN_TRAIN_ROWS = 200     # need at least this many labeled rows to start training
BATCH_SIZE = 256

def build_feature_df():
    if len(snapshot_cache) < 60:
        return None
    df = pd.DataFrame(list(snapshot_cache))
    # sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    # returns over 1/3/5 samples (we don't know sample frequency; use rows as proxy)
    df["r1"] = df["mid"].pct_change(1)
    df["r3"] = df["mid"].pct_change(3)
    df["r5"] = df["mid"].pct_change(5)
    df["d_micro"] = df["micro"].diff()
    df["d_ofi"] = df["ofi"].diff()
    df["d_spread"] = df["spread"].diff()
    df["d_slope"] = df["slope"].diff()
    df["d_vpin"] = df["vpin"].diff()
    df = df.dropna().reset_index(drop=True)
    return df

def label_future(df, horizon_seconds=300):
    # For each row, find the first future row with timestamp >= t + horizon_seconds
    df = df.copy()
    future_price = []
    timestamps = df["timestamp"].values
    mids = df["mid"].values
    n = len(df)
    for i in range(n):
        target_ts = timestamps[i] + horizon_seconds
        # find index j where timestamps[j] >= target_ts
        j = np.searchsorted(timestamps, target_ts, side='left')
        if j >= n:
            future_price.append(np.nan)
        else:
            future_price.append(mids[j])
    df["future_mid"] = future_price
    df["future_return_5m"] = (df["future_mid"] - df["mid"]) / df["mid"]
    # map to class: 1 = up, 0 = neutral/down (binary). Will treat neutral as 0.
    # You can change thresholds
    up_thresh = 0.0005    # 0.05% move
    down_thresh = -0.0005
    def to_class(x):
        if np.isnan(x):
            return np.nan
        if x > up_thresh:
            return 1
        elif x < down_thresh:
            return -1
        else:
            return 0
    df["label"] = df["future_return_5m"].apply(to_class)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    return df

def prepare_training_batches(df):
    # Keep only rows with label in {-1,0,1}
    df = df[df["label"].isin([-1,0,1])].reset_index(drop=True)
    if len(df) < MIN_TRAIN_ROWS:
        return None
    # We'll convert to a binary classification of up vs not-up for SGDClassifier
    # y = 1 if label == 1, else 0
    df["y"] = (df["label"] == 1).astype(int)
    features = []
    for c in ["mid","micro","ofi","spread","bid_pressure","ask_pressure","pressure_ratio",
              "slope","imbalance","vpin","volatility","r1","r3","r5","d_micro","d_ofi","d_spread","d_slope","d_vpin"]:
        if c in df.columns:
            features.append(c)
        else:
            df[c] = 0.0
            features.append(c)
    X = df[features].values
    y = df["y"].values
    return X, y, features

def online_train():
    global clf_initialized, scaler, clf
    df = build_feature_df()
    if df is None:
        return
    df = label_future(df, horizon_seconds=300)
    res = prepare_training_batches(df)
    if res is None:
        return
    X, y, features = res

    # partial fit in mini-batches
    if not clf_initialized:
        # initial scaler fit and classifier partial_fit with classes [0,1]
        scaler.partial_fit(X)
        Xs = scaler.transform(X)
        # first partial_fit can take whole X (ok)
        clf.partial_fit(Xs, y, classes=np.array([0,1]))
        clf_initialized = True
    else:
        # update scaler then classifier by chunks
        scaler.partial_fit(X)
        # shuffle and partial_fit
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]
        for i in range(0, len(X), BATCH_SIZE):
            xb = X[i:i+BATCH_SIZE]
            yb = y[i:i+BATCH_SIZE]
            Xsb = scaler.transform(xb)
            clf.partial_fit(Xsb, yb)

def predict_5min(latest_snapshot):
    # returns probability of up
    if not clf_initialized:
        return None
    df = build_feature_df()
    if df is None or len(df) == 0:
        return None
    # build feature vector from latest_snapshot
    feat = []
    for c in ["mid","micro","ofi","spread","bid_pressure","ask_pressure","pressure_ratio",
              "slope","imbalance","vpin","volatility","r1","r3","r5","d_micro","d_ofi","d_spread","d_slope","d_vpin"]:
        if c in latest_snapshot:
            feat.append(latest_snapshot[c])
        else:
            # compute from last two rows in snapshot_cache if needed
            feat.append(0.0)
    Xv = np.array(feat).reshape(1, -1)
    Xs = scaler.transform(Xv)
    prob_up = clf.predict_proba(Xs)[0][1]
    pred_class = clf.predict(Xs)[0]
    label = "BULLISH 5m 📈" if pred_class == 1 else "NOT-BULLISH 5m"
    return {"prob_up": float(prob_up), "pred_class": int(pred_class), "label": label}

# -------------------------------
# Main WebSocket depth stream (merged)
# -------------------------------
async def depth_stream():
    print("🔵 Merged Trend Model + HFT Indicators (with 5-min online prediction)\n")
    update_times = deque(maxlen=200)

    async with websockets.connect(WS_URL) as ws:
        async for message in ws:
            try:
                msg = json.loads(message)
            except Exception:
                continue

            # timestamp
            now = datetime.utcnow()
            update_times.append(now.timestamp())

            # update orderbook
            for price, qty in msg.get("b", []):
                p = float(price); q = float(qty)
                if q == 0: bids.pop(p, None)
                else: bids[p] = q
            for price, qty in msg.get("a", []):
                p = float(price); q = float(qty)
                if q == 0: asks.pop(p, None)
                else: asks[p] = q

            if not bids or not asks:
                continue

            # compute indicators
            try:
                best_bid = max(bids.keys())
                best_ask = min(asks.keys())
            except ValueError:
                continue

            midprice = (best_bid + best_ask) / 2.0
            # update price history for short-term predictors
            price_history.append(midprice)

            # HFT metrics
            micro = microprice_indicator()
            ofi = order_flow_imbalance()
            spread = spread_indicator()
            bp, ap, pratio = pressure_indicator()
            slope = orderbook_slope()
            imb = inventory_imbalance()
            vpin = vpin_indicator(midprice)
            vol = short_term_volatility(midprice)
            liquidity = liquidity_shock()
            stress = market_stress_index()

            hft = {
                "microprice": micro,
                "ofi": ofi,
                "spread": spread,
                "bid_pressure": bp,
                "ask_pressure": ap,
                "pressure_ratio": pratio,
                "orderbook_slope": slope,
                "imbalance": imb,
                "vpin": vpin,
                "volatility": vol,
                "liquidity_shock": liquidity,
                "stress": stress
            }

            # get last traded price via REST (public ticker)
            last_price = fetch_last_price(SYMBOL_REST)
            if last_price is None:
                last_price = midprice

            # predictions from 5 methods
            preds = [
                predict_lr(list(price_history)) if len(price_history) >= 2 else last_price,
                predict_hma(list(price_history)),
                predict_kalman(list(price_history)),
                predict_cwma(list(price_history)),
                predict_dma(list(price_history))
            ]
            trend = merge_signals(preds, last_price)

            # create snapshot & add to cache for 5-min model
            snapshot = make_snapshot(
                mid=midprice,
                micro=micro,
                ofi=ofi,
                spread=spread,
                bid_pressure=bp,
                ask_pressure=ap,
                pressure_ratio=pratio if pratio is not None else 0.0,
                slope=slope,
                imbalance=imb,
                vpin=vpin if vpin is not None else 0.0,
                volatility=vol if vol is not None else 0.0
            )
            # add engineering features used by online model
            # We add placeholders for r1,r3,r5 and diffs — build_feature_df will compute them.
            snapshot_cache.append(snapshot)
            # prune older than CACHE_SECONDS
            cutoff = datetime.utcnow().timestamp() - CACHE_SECONDS
            while len(snapshot_cache) and snapshot_cache[0]["timestamp"] < cutoff:
                snapshot_cache.popleft()

            # Train the online model occasionally (cheap)
            # We'll call training every N updates
            if len(snapshot_cache) % 50 == 0:
                online_train()

            # inference
            latest_for_pred = None
            # create latest_snapshot dict with same keys as feature builder expects
            df_tmp = build_feature_df()
            if df_tmp is not None and len(df_tmp) > 0:
                latest_row = df_tmp.iloc[-1].to_dict()
                latest_for_pred = latest_row

            pred5 = predict_5min(latest_for_pred) if latest_for_pred is not None else None

            # -----------------------------
            # PRINT MERGED OUTPUT
            # -----------------------------
            print("\n⏱", datetime.utcnow(), "UTC")
            print("============================================")
            print("🔶 TREND SIGNAL (Merged ML Models):", trend)
            if pred5:
                print("🔮 5-MIN PREDICTION:", pred5["label"], f"(prob_up={pred5['prob_up']:.4f})")
            else:
                print("🔮 5-MIN PREDICTION: warming up / not enough data")
            print("============================================")
            # print HFT indicators
            print(f"Microprice:      {micro}")
            print(f"OFI:             {ofi}")
            print(f"Spread:          {spread}")
            print(f"Bid Pressure:    {bp}")
            print(f"Ask Pressure:    {ap}")
            print(f"Pressure Ratio:  {pratio}")
            print(f"Orderbook Slope: {slope}")
            print(f"Imbalance:       {imb}")
            print(f"VPIN:            {vpin}")
            print(f"Volatility:      {vol}")
            print(f"Liquidity Shock: {liquidity}")
            print(f"Stress Index:    {stress}")
            print("--------------------------------------------------")

# -------------------------------
# Run (COLAB friendly)
# -------------------------------
# In a notebook cell you can run:
# await depth_stream()
#
# or run in asyncio event loop in a script:
# asyncio.run(depth_stream())

# For convenience in a notebook use `await` at top-level:
await depth_stream()
