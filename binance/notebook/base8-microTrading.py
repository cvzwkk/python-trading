
# =========================
# INSTALL DEPENDENCIES
# =========================
!pip install river
!pip install pandas==2.2.2
!pip install python-binance
!pip install websockets
!pip install aiohttp
!pip install nest_asyncio
!pip install matplotlib
!pip install seaborn
!pip install pykalman
# =========================
# IMPORTS
# =========================
import asyncio
import websockets
import json
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
window_size = 30  # trend model window
cache_window = 300  # last 5 minutes for River ML
price_history = []

# Simulation parameters
USD_balance = 10000.0  # starting USD
BTC_balance = 0.0
trade_size_pct = 0.5  # trade 50% of available balance
trade_cache = []  # store executed trades
last_signal = None

# Micro-scalping parameters
micro_trade_fraction = 0.001  # 0.1% of balance per micro-trade
micro_spread_target = 0.0003  # 0.03% spread to capture

# Binance client
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
    deque(maxlen=50),
)
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


# =========================
# TREND PREDICTION MODELS
# =========================
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
        weights = np.arange(1, n + 1)
        return np.sum(arr[-n:] * weights) / weights.sum()

    half = period // 2
    sqrt_len = int(np.sqrt(period))
    wma_half = wma(np.array(prices), half)
    wma_full = wma(np.array(prices), period)
    raw_hma = 2 * wma_half - wma_full
    return float(wma(np.array([raw_hma]), sqrt_len))


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
    return float(np.average(prices, weights=np.full(len(prices), weight)))


def predict_dma(prices, displacement=3):
    if len(prices) <= displacement:
        return prices[-1]
    return np.mean(prices[-displacement:])


# ===== Exotic Averages =====
def predict_ema(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode="valid")[0])


def predict_tema(prices, period=10):
    if len(prices) < period * 3:
        return prices[-1]
    ema1 = predict_ema(prices, period)
    ema2 = predict_ema(
        [predict_ema(prices[: i + 1], period) for i in range(len(prices))], period
    )
    ema3 = predict_ema(
        [
            predict_ema(
                [predict_ema(prices[: i + 1], period) for i in range(j + 1)], period
            )
            for j in range(len(prices))
        ],
        period,
    )
    return float(3 * ema1 - 3 * ema2 + ema3)


def predict_wma(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    weights = np.arange(1, period + 1)
    return float(np.dot(prices[-period:], weights) / weights.sum())


def predict_smma(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    smma = np.mean(prices[:period])
    for p in prices[period:]:
        smma = (smma * (period - 1) + p) / period
    return float(smma)


# ===== Merge Signals =====
def merge_signals(preds, last_price, weights=None):
    if weights is None:
        weights = np.ones(len(preds))
    signals = [
        w * (1 if p > last_price else -1 if p < last_price else 0)
        for w, p in zip(weights, preds)
    ]
    score = sum(signals)
    if score > 0:
        return "BULLISH 📈"
    elif score < 0:
        return "BEARISH 📉"
    return "NEUTRAL ➖"


# =========================
# HFT & FPGA INDICATORS
# =========================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid] + asks[best_ask]
    return (best_bid * asks[best_ask] + best_ask * bids[best_bid]) / w


def spread_indicator():
    return min(asks.keys()) - max(bids.keys())


def order_flow_imbalance():
    global last_best_bid, last_best_ask
    best_bid, max_best_ask = max(bids.keys()), min(asks.keys())
    ofi = 0
    if last_best_bid is not None:
        ofi += best_bid - last_best_bid
    if last_best_ask is not None:
        ofi += last_best_ask - max_best_ask
    last_best_bid, last_best_ask = best_bid, max_best_ask
    ofi_window.append(ofi)
    return ofi


def pressure_indicator(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0, 0, None
    bid_pressure = sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_pressure = sum([asks[top_asks[i]] for i in range(available_levels)])
    ratio = bid_pressure / ask_pressure if ask_pressure > 0 else None
    return bid_pressure, ask_pressure, ratio


def orderbook_slope(depth=10):
    prices = sorted(list(bids.keys()) + list(asks.keys()))
    quantities = [bids.get(p, asks.get(p, 0)) for p in prices]
    if len(prices) < 3:
        return 0
    return np.polyfit(prices, quantities, 1)[0]


def inventory_imbalance(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0
    B = sum([bids[top_bids[i]] for i in range(available_levels)])
    A = sum([asks[top_asks[i]] for i in range(available_levels)])
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
    return np.std(np.diff(vol_window))


def liquidity_shock():
    spread = spread_indicator()
    return (
        spread > 1.5 * np.mean([abs(x) for x in vol_window])
        if len(vol_window) > 10
        else None
    )


# FPGA-style features
def weighted_imbalance(levels=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(levels, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0
    imbalance = 0
    weight_sum = 0
    for i in range(available_levels):
        w = 1 / (i + 1)
        b_qty = bids.get(top_bids[i], 0)
        a_qty = asks.get(top_asks[i], 0)
        imbalance += w * (b_qty - a_qty)
        weight_sum += w * (b_qty + a_qty)
    return imbalance / weight_sum if weight_sum != 0 else 0


def rolling_ofi_sum():
    return sum(ofi_window)


def micro_momentum(price):
    micro_window.append(price)
    if len(micro_window) < 2:
        return 0
    return micro_window[-1] - micro_window[0]


def cancellation_ratio(msg):
    cancels = sum(1 for p, q in msg.get("b", []) if q == 0) + sum(
        1 for p, q in msg.get("a", []) if q == 0
    )
    cancel_window.append(cancels)
    return np.mean(cancel_window)


def price_skew(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0
    bid_vol = sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_vol = sum([asks[top_asks[i]] for i in range(available_levels)])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)


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
    y_class = {1: "BULLISH 📈", -1: "BEARISH 📉", 0: "NEUTRAL ➖"}
    online_log.learn_one(x_scaled, trend)
    return y_pred, y_class[trend]


# =========================
# MICRO-SCALPING STRATEGY
# =========================
def micro_scalp(midprice, best_bid, best_ask, USD_balance, BTC_balance, trade_cache):
    signal_alert = None
    # Only trade if spread >= target
    if best_ask - best_bid >= midprice * micro_spread_target:
        usd_to_spend = USD_balance * micro_trade_fraction
        btc_to_buy = usd_to_spend / best_bid
        if usd_to_spend > 0:
            USD_balance -= usd_to_spend
            BTC_balance += btc_to_buy
            # Immediate micro-sell
            usd_gained = btc_to_buy * best_ask
            BTC_balance -= btc_to_buy
            USD_balance += usd_gained
            trade_cache.append(
                {
                    "type": "MICRO",
                    "buy": best_bid,
                    "sell": best_ask,
                    "btc": btc_to_buy,
                    "usd": usd_to_spend,
                    "time": datetime.utcnow(),
                }
            )
            signal_alert = f"⚡ MICRO-SCALP! Bought {btc_to_buy:.6f} BTC @ {best_bid:.2f} | Sold @ {best_ask:.2f} | Profit: {(usd_gained-usd_to_spend):.2f} USD"
    return USD_balance, BTC_balance, signal_alert


# =========================
# LIVE STREAM LOOP
# =========================
# =========================
# LIVE STREAM LOOP
# =========================
async def depth_stream():
    global USD_balance, BTC_balance, trade_cache
    print("🔵 High-Frequency Micro-Scalping Engine Simulation\n")

    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # Update orderbook
            for price, qty in msg["b"]:
                p, q = float(price), float(qty)
                if q == 0:
                    bids.pop(p, None)
                else:
                    bids[p] = q
            for price, qty in msg["a"]:
                p, q = float(price), float(qty)
                if q == 0:
                    asks.pop(p, None)
                else:
                    asks[p] = q
            if not bids or not asks:
                continue

            best_bid, max_best_ask = max(bids.keys()), min(asks.keys())
            midprice = (best_bid + max_best_ask) / 2

            # HFT features
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
                "liquidity_shock": liquidity_shock(),
            }

            # FPGA features
            w_imb = weighted_imbalance()
            r_ofi = rolling_ofi_sum()
            micro_mom = micro_momentum(midprice)
            cancel_r = cancellation_ratio(msg)
            p_skew = price_skew()
            fpga_features = {
                "weighted_imbalance": (w_imb, trend_signal(w_imb, 0)),
                "rolling_ofi": (r_ofi, trend_signal(r_ofi, 0)),
                "micro_momentum": (micro_mom, trend_signal(micro_mom, 0)),
                "cancel_ratio": (cancel_r, trend_signal(cancel_r, 0)),
                "price_skew": (p_skew, trend_signal(p_skew, 0)),
            }

            # River prediction
            next_pred, next_trend = update_river_models(
                midprice, {k: v[0] for k, v in fpga_features.items()}
            )
            snapshot_cache.append(
                {
                    "midprice": midprice,
                    **hft_features,
                    **{k: v[0] for k, v in fpga_features.items()},
                }
            )

            # =========================
            # MICRO-SCALPING ONLY
            # =========================
            USD_balance, BTC_balance, micro_signal = micro_scalp(
                midprice, best_bid, max_best_ask, USD_balance, BTC_balance, trade_cache
            )

            total_equity = USD_balance + BTC_balance * midprice

            # =========================
            # PRINT
            # =========================
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ HFT Indicators:")
            for k, v in hft_features.items():
                print(f"   {k:18}: {v}")
            print("----------------------------------------")
            print("⭐ FPGA Features:")
            for k, (val, sig) in fpga_features.items():
                print(f"   {k:18}: {val} | {sig}")
            print("----------------------------------------")
            print("⭐ River Online Prediction:", f"{next_pred:.2f} | {next_trend}")

            # Neat Micro-Scalp formatting
            if micro_signal:
                last_trade = trade_cache[-1]
                profit = (
                    last_trade["usd"]
                    * (last_trade["sell"] - last_trade["buy"])
                    / last_trade["buy"]
                )
                print("----------------------------------------")
                print("⚡SCALP!")
                print(
                    f" Bought {last_trade['buy']:.2f} USDT @ {last_trade['btc']:.6f} BTC"
                )
                print(f" Sold @ {last_trade['sell']:.2f} USDT")
                print(f" Profit: {profit:.2f} USDT")
                print("----------------------------------------")

            print(f"💹 Balance: USD {USD_balance:.2f} | BTC {BTC_balance:.6f} ")
            print(f"💰Total Equity: {total_equity:.2f} USDT")
            print(f"💲 Latest Price: {midprice:.2f} USDT")
            print("----------------------------------------")


# =========================
# RUN
# =========================
await depth_stream()# Micro-scalping parameters
micro_trade_fraction = 0.001  # 0.1% of balance per micro-trade
micro_spread_target = 0.0003  # 0.03% spread to capture

# Binance client
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
    deque(maxlen=50),
)
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


# =========================
# TREND PREDICTION MODELS
# =========================
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
        weights = np.arange(1, n + 1)
        return np.sum(arr[-n:] * weights) / weights.sum()

    half = period // 2
    sqrt_len = int(np.sqrt(period))
    wma_half = wma(np.array(prices), half)
    wma_full = wma(np.array(prices), period)
    raw_hma = 2 * wma_half - wma_full
    return float(wma(np.array([raw_hma]), sqrt_len))


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
    return float(np.average(prices, weights=np.full(len(prices), weight)))


def predict_dma(prices, displacement=3):
    if len(prices) <= displacement:
        return prices[-1]
    return np.mean(prices[-displacement:])


# ===== Exotic Averages =====
def predict_ema(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    weights = np.exp(np.linspace(-1.0, 0.0, period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode="valid")[0])


def predict_tema(prices, period=10):
    if len(prices) < period * 3:
        return prices[-1]
    ema1 = predict_ema(prices, period)
    ema2 = predict_ema(
        [predict_ema(prices[: i + 1], period) for i in range(len(prices))], period
    )
    ema3 = predict_ema(
        [
            predict_ema(
                [predict_ema(prices[: i + 1], period) for i in range(j + 1)], period
            )
            for j in range(len(prices))
        ],
        period,
    )
    return float(3 * ema1 - 3 * ema2 + ema3)


def predict_wma(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    weights = np.arange(1, period + 1)
    return float(np.dot(prices[-period:], weights) / weights.sum())


def predict_smma(prices, period=10):
    if len(prices) < period:
        return prices[-1]
    smma = np.mean(prices[:period])
    for p in prices[period:]:
        smma = (smma * (period - 1) + p) / period
    return float(smma)


# ===== Merge Signals =====
def merge_signals(preds, last_price, weights=None):
    if weights is None:
        weights = np.ones(len(preds))
    signals = [
        w * (1 if p > last_price else -1 if p < last_price else 0)
        for w, p in zip(weights, preds)
    ]
    score = sum(signals)
    if score > 0:
        return "BULLISH 📈"
    elif score < 0:
        return "BEARISH 📉"
    return "NEUTRAL ➖"


# =========================
# HFT & FPGA INDICATORS
# =========================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid] + asks[best_ask]
    return (best_bid * asks[best_ask] + best_ask * bids[best_bid]) / w


def spread_indicator():
    return min(asks.keys()) - max(bids.keys())


def order_flow_imbalance():
    global last_best_bid, last_best_ask
    best_bid, max_best_ask = max(bids.keys()), min(asks.keys())
    ofi = 0
    if last_best_bid is not None:
        ofi += best_bid - last_best_bid
    if last_best_ask is not None:
        ofi += last_best_ask - max_best_ask
    last_best_bid, last_best_ask = best_bid, max_best_ask
    ofi_window.append(ofi)
    return ofi


def pressure_indicator(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0, 0, None
    bid_pressure = sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_pressure = sum([asks[top_asks[i]] for i in range(available_levels)])
    ratio = bid_pressure / ask_pressure if ask_pressure > 0 else None
    return bid_pressure, ask_pressure, ratio


def orderbook_slope(depth=10):
    prices = sorted(list(bids.keys()) + list(asks.keys()))
    quantities = [bids.get(p, asks.get(p, 0)) for p in prices]
    if len(prices) < 3:
        return 0
    return np.polyfit(prices, quantities, 1)[0]


def inventory_imbalance(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0
    B = sum([bids[top_bids[i]] for i in range(available_levels)])
    A = sum([asks[top_asks[i]] for i in range(available_levels)])
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
    return np.std(np.diff(vol_window))


def liquidity_shock():
    spread = spread_indicator()
    return (
        spread > 1.5 * np.mean([abs(x) for x in vol_window])
        if len(vol_window) > 10
        else None
    )


# FPGA-style features
def weighted_imbalance(levels=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(levels, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0
    imbalance = 0
    weight_sum = 0
    for i in range(available_levels):
        w = 1 / (i + 1)
        b_qty = bids.get(top_bids[i], 0)
        a_qty = asks.get(top_asks[i], 0)
        imbalance += w * (b_qty - a_qty)
        weight_sum += w * (b_qty + a_qty)
    return imbalance / weight_sum if weight_sum != 0 else 0


def rolling_ofi_sum():
    return sum(ofi_window)


def micro_momentum(price):
    micro_window.append(price)
    if len(micro_window) < 2:
        return 0
    return micro_window[-1] - micro_window[0]


def cancellation_ratio(msg):
    cancels = sum(1 for p, q in msg.get("b", []) if q == 0) + sum(
        1 for p, q in msg.get("a", []) if q == 0
    )
    cancel_window.append(cancels)
    return np.mean(cancel_window)


def price_skew(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth, len(top_bids), len(top_asks))
    if available_levels == 0:
        return 0
    bid_vol = sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_vol = sum([asks[top_asks[i]] for i in range(available_levels)])
    return (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)


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
    y_class = {1: "BULLISH 📈", -1: "BEARISH 📉", 0: "NEUTRAL ➖"}
    online_log.learn_one(x_scaled, trend)
    return y_pred, y_class[trend]


# =========================
# MICRO-SCALPING STRATEGY
# =========================
def micro_scalp(midprice, best_bid, best_ask, USD_balance, BTC_balance, trade_cache):
    signal_alert = None
    # Only trade if spread >= target
    if best_ask - best_bid >= midprice * micro_spread_target:
        usd_to_spend = USD_balance * micro_trade_fraction
        btc_to_buy = usd_to_spend / best_bid
        if usd_to_spend > 0:
            USD_balance -= usd_to_spend
            BTC_balance += btc_to_buy
            # Immediate micro-sell
            usd_gained = btc_to_buy * best_ask
            BTC_balance -= btc_to_buy
            USD_balance += usd_gained
            trade_cache.append(
                {
                    "type": "MICRO",
                    "buy": best_bid,
                    "sell": best_ask,
                    "btc": btc_to_buy,
                    "usd": usd_to_spend,
                    "time": datetime.utcnow(),
                }
            )
            signal_alert = f"⚡ MICRO-SCALP! Bought {btc_to_buy:.6f} BTC @ {best_bid:.2f} | Sold @ {best_ask:.2f} | Profit: {(usd_gained-usd_to_spend):.2f} USD"
    return USD_balance, BTC_balance, signal_alert


# =========================
# LIVE STREAM LOOP
# =========================
# =========================
# LIVE STREAM LOOP
# =========================
async def depth_stream():
    global USD_balance, BTC_balance, trade_cache
    print("🔵 High-Frequency Micro-Scalping Engine Simulation\n")

    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # Update orderbook
            for price, qty in msg["b"]:
                p, q = float(price), float(qty)
                if q == 0:
                    bids.pop(p, None)
                else:
                    bids[p] = q
            for price, qty in msg["a"]:
                p, q = float(price), float(qty)
                if q == 0:
                    asks.pop(p, None)
                else:
                    asks[p] = q
            if not bids or not asks:
                continue

            best_bid, max_best_ask = max(bids.keys()), min(asks.keys())
            midprice = (best_bid + max_best_ask) / 2

            # HFT features
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
                "liquidity_shock": liquidity_shock(),
            }

            # FPGA features
            w_imb = weighted_imbalance()
            r_ofi = rolling_ofi_sum()
            micro_mom = micro_momentum(midprice)
            cancel_r = cancellation_ratio(msg)
            p_skew = price_skew()
            fpga_features = {
                "weighted_imbalance": (w_imb, trend_signal(w_imb, 0)),
                "rolling_ofi": (r_ofi, trend_signal(r_ofi, 0)),
                "micro_momentum": (micro_mom, trend_signal(micro_mom, 0)),
                "cancel_ratio": (cancel_r, trend_signal(cancel_r, 0)),
                "price_skew": (p_skew, trend_signal(p_skew, 0)),
            }

            # River prediction
            next_pred, next_trend = update_river_models(
                midprice, {k: v[0] for k, v in fpga_features.items()}
            )
            snapshot_cache.append(
                {
                    "midprice": midprice,
                    **hft_features,
                    **{k: v[0] for k, v in fpga_features.items()},
                }
            )

            # =========================
            # MICRO-SCALPING ONLY
            # =========================
            USD_balance, BTC_balance, micro_signal = micro_scalp(
                midprice, best_bid, max_best_ask, USD_balance, BTC_balance, trade_cache
            )

            total_equity = USD_balance + BTC_balance * midprice

            # =========================
            # PRINT
            # =========================
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ HFT Indicators:")
            for k, v in hft_features.items():
                print(f"   {k:18}: {v}")
            print("----------------------------------------")
            print("⭐ FPGA Features:")
            for k, (val, sig) in fpga_features.items():
                print(f"   {k:18}: {val} | {sig}")
            print("----------------------------------------")
            print("⭐ River Online Prediction:", f"{next_pred:.2f} | {next_trend}")

            # Neat Micro-Scalp formatting
            if micro_signal:
                last_trade = trade_cache[-1]
                profit = (
                    last_trade["usd"]
                    * (last_trade["sell"] - last_trade["buy"])
                    / last_trade["buy"]
                )
                print("----------------------------------------")
                print("⚡SCALP!")
                print(
                    f" Bought {last_trade['buy']:.2f} USDT @ {last_trade['btc']:.6f} BTC"
                )
                print(f" Sold @ {last_trade['sell']:.2f} USDT")
                print(f" Profit: {profit:.2f} USDT")
                print("----------------------------------------")

            print(f"💹 Balance: USD {USD_balance:.2f} | BTC {BTC_balance:.6f} ")
            print(f"💰Total Equity: {total_equity:.2f} USDT")
            print("----------------------------------------")


# =========================
# RUN
# =========================
await depth_stream()
# Micro-scalping parameters
micro_trade_fraction = 0.001   # 0.1% of balance per micro-trade
micro_spread_target = 0.0003   # 0.03% spread to capture

# Binance client
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

# ===== Exotic Averages =====
def predict_ema(prices, period=10):
    if len(prices) < period: return prices[-1]
    weights = np.exp(np.linspace(-1., 0., period))
    weights /= weights.sum()
    return float(np.convolve(prices[-period:], weights, mode='valid')[0])

def predict_tema(prices, period=10):
    if len(prices) < period*3: return prices[-1]
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
        smma = (smma*(period-1) + p)/period
    return float(smma)

# ===== Merge Signals =====
def merge_signals(preds, last_price, weights=None):
    if weights is None:
        weights = np.ones(len(preds))
    signals = [w*(1 if p>last_price else -1 if p<last_price else 0) for w,p in zip(weights,preds)]
    score = sum(signals)
    if score>0: return "BULLISH 📈"
    elif score<0: return "BEARISH 📉"
    return "NEUTRAL ➖"

# =========================
# HFT & FPGA INDICATORS
# =========================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid]+asks[best_ask]
    return (best_bid*asks[best_ask]+best_ask*bids[best_bid])/w

def spread_indicator(): return min(asks.keys()) - max(bids.keys())

def order_flow_imbalance():
    global last_best_bid,last_best_ask
    best_bid,max_best_ask = max(bids.keys()),min(asks.keys())
    ofi = 0
    if last_best_bid is not None: ofi += best_bid-last_best_bid
    if last_best_ask is not None: ofi += last_best_ask-max_best_ask
    last_best_bid,last_best_ask = best_bid,max_best_ask
    ofi_window.append(ofi)
    return ofi

def pressure_indicator(depth=5):
    top_bids = sorted(bids.keys(), reverse=True)
    top_asks = sorted(asks.keys())
    available_levels = min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0,0,None
    bid_pressure=sum([bids[top_bids[i]] for i in range(available_levels)])
    ask_pressure=sum([asks[top_asks[i]] for i in range(available_levels)])
    ratio=bid_pressure/ask_pressure if ask_pressure>0 else None
    return bid_pressure,ask_pressure,ratio

def orderbook_slope(depth=10):
    prices = sorted(list(bids.keys())+list(asks.keys()))
    quantities=[bids.get(p,asks.get(p,0)) for p in prices]
    if len(prices)<3: return 0
    return np.polyfit(prices,quantities,1)[0]

def inventory_imbalance(depth=5):
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(depth,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    B=sum([bids[top_bids[i]] for i in range(available_levels)])
    A=sum([asks[top_asks[i]] for i in range(available_levels)])
    return (B-A)/(B+A+1e-9)

def vpin_indicator(price):
    vpin_window.append(price)
    if len(vpin_window)<vpin_window.maxlen: return None
    returns=np.diff(vpin_window)
    buy_volume=np.sum(returns>0)
    sell_volume=np.sum(returns<0)
    return abs(buy_volume-sell_volume)/(buy_volume+sell_volume+1e-9)

def short_term_volatility(price):
    vol_window.append(price)
    if len(vol_window)<vol_window.maxlen: return None
    return np.std(np.diff(vol_window))

def liquidity_shock():
    spread = spread_indicator()
    return spread > 1.5*np.mean([abs(x) for x in vol_window]) if len(vol_window)>10 else None

# FPGA-style features
def weighted_imbalance(levels=5):
    top_bids=sorted(bids.keys(),reverse=True)
    top_asks=sorted(asks.keys())
    available_levels=min(levels,len(top_bids),len(top_asks))
    if available_levels==0: return 0
    imbalance=0; weight_sum=0
    for i in range(available_levels):
        w=1/(i+1)
        b_qty=bids.get(top_bids[i],0)
        a_qty=asks.get(top_asks[i],0)
        imbalance+=w*(b_qty-a_qty)
        weight_sum+=w*(b_qty+a_qty)
    return imbalance/weight_sum if weight_sum!=0 else 0

def rolling_ofi_sum(): return sum(ofi_window)
def micro_momentum(price):
    micro_window.append(price)
    if len(micro_window)<2: return 0
    return micro_window[-1]-micro_window[0]

def cancellation_ratio(msg):
    cancels=sum(1 for p,q in msg.get("b",[]) if q==0)+sum(1 for p,q in msg.get("a",[]) if q==0)
    cancel_window.append(cancels)
    return np.mean(cancel_window)

def price_skew(depth=5):
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
# MICRO-SCALPING STRATEGY
# =========================
def micro_scalp(midprice, best_bid, best_ask, USD_balance, BTC_balance, trade_cache):
    signal_alert = None
    # Only trade if spread >= target
    if best_ask - best_bid >= midprice * micro_spread_target:
        usd_to_spend = USD_balance * micro_trade_fraction
        btc_to_buy = usd_to_spend / best_bid
        if usd_to_spend > 0:
            USD_balance -= usd_to_spend
            BTC_balance += btc_to_buy
            # Immediate micro-sell
            usd_gained = btc_to_buy * best_ask
            BTC_balance -= btc_to_buy
            USD_balance += usd_gained
            trade_cache.append({"type":"MICRO","buy":best_bid,"sell":best_ask,
                                "btc":btc_to_buy,"usd":usd_to_spend,"time":datetime.utcnow()})
            signal_alert = f"⚡ MICRO-SCALP! Bought {btc_to_buy:.6f} BTC @ {best_bid:.2f} | Sold @ {best_ask:.2f} | Profit: {(usd_gained-usd_to_spend):.2f} USD"
    return USD_balance, BTC_balance, signal_alert

# =========================
# LIVE STREAM LOOP
# =========================
# =========================
# RUN THE ASYNC LOOP
# =========================
async def depth_stream():
    global USD_balance, BTC_balance, trade_cache
    print("🔵 High-Frequency Micro-Scalping Engine Simulation\n")
    
    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)
            
            # Update orderbook
            for price, qty in msg["b"]:
                p,q = float(price), float(qty)
                if q==0: bids.pop(p,None)
                else: bids[p]=q
            for price, qty in msg["a"]:
                p,q = float(price), float(qty)
                if q==0: asks.pop(p,None)
                else: asks[p]=q
            if not bids or not asks: continue

            best_bid,max_best_ask = max(bids.keys()), min(asks.keys())
            midprice = (best_bid + max_best_ask)/2

            # HFT features
            ofi = order_flow_imbalance()
            bid_p, ask_p, ratio = pressure_indicator()
            hft_features={
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

            # FPGA features
            w_imb = weighted_imbalance()
            r_ofi = rolling_ofi_sum()
            micro_mom = micro_momentum(midprice)
            cancel_r = cancellation_ratio(msg)
            p_skew = price_skew()
            fpga_features = {
                "weighted_imbalance": (w_imb, trend_signal(w_imb,0)),
                "rolling_ofi": (r_ofi, trend_signal(r_ofi,0)),
                "micro_momentum": (micro_mom, trend_signal(micro_mom,0)),
                "cancel_ratio": (cancel_r, trend_signal(cancel_r,0)),
                "price_skew": (p_skew, trend_signal(p_skew,0))
            }

            # River prediction
            next_pred,next_trend = update_river_models(midprice,{k:v[0] for k,v in fpga_features.items()})
            snapshot_cache.append({"midprice": midprice, **hft_features, **{k:v[0] for k,v in fpga_features.items()}})

            # =========================
            # MICRO-SCALPING ONLY
            # =========================
            USD_balance, BTC_balance, micro_signal = micro_scalp(midprice, best_bid, max_best_ask, USD_balance, BTC_balance, trade_cache)

            # Total equity
            total_equity = USD_balance + BTC_balance*midprice

            # =========================
            # PRINT
            # =========================
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ HFT Indicators:")
            for k, v in hft_features.items():
                print(f"   {k:18}: {v}")
            
            print("⭐ FPGA Features:")
            for k, (val, sig) in fpga_features.items():
                print(f"   {k:18}: {val} | {sig}")
            
            print("⭐ River Online Prediction:", f"{next_pred:.2f} | {next_trend}")
            
            if micro_signal:
                print(micro_signal)
            
            print(f"💹 Simulated Balance: USD {USD_balance:.2f} | BTC {BTC_balance:.6f} | Total Equity: {total_equity:.2f} USD")
            print("------------------------------------------------------------")

# =========================
# RUN
# =========================
await depth_stream()
