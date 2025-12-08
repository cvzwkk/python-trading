
# =========================
# INSTALL DEPENDENCIES
# =========================
#!pip install python-binance pykalman websockets nest_asyncio river scipy --quiet

# =========================
# IMPORTS
# =========================
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

nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
interval = 1  # seconds for price fetch
window_size = 30  # trend model window
cache_window = 300  # last 5 minutes for River ML
price_history = []

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
# MULTI-TRADING SYSTEM
# =========================
balance = 1000.0
positions = {}  # key: strategy_name, value: {"type": "LONG"/"SHORT", "entry": price}
trade_log = []

# =========================
# UTILITY FUNCTIONS
# =========================
def trend_signal(pred, last_price):
    if pred > last_price: return "BULLISH 📈"
    elif pred < last_price: return "BEARISH 📉"
    return "NEUTRAL ➖"

def fetch_last_price(symbol):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker['price'])

def merge_signals(preds, last_price):
    signals = [1 if p>last_price else -1 if p<last_price else 0 for p in preds]
    score = sum(signals)
    if score>0: return "BULLISH 📈"
    elif score<0: return "BEARISH 📉"
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
# HFT INDICATORS
# =========================
def microprice_indicator():
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    w = bids[best_bid]+asks[best_ask]
    return (best_bid*asks[best_ask]+best_ask*bids[best_bid])/w

def spread_indicator():
    return min(asks.keys()) - max(bids.keys())

def order_flow_imbalance():
    global last_best_bid,last_best_ask
    best_bid = max(bids.keys())
    best_ask = min(asks.keys())
    ofi = 0
    if last_best_bid is not None: ofi += best_bid-last_best_bid
    if last_best_ask is not None: ofi += last_best_ask-best_ask
    last_best_bid,last_best_ask = best_bid,best_ask
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

# =========================
# FPGA-STYLE STREAMING FEATURES
# =========================
def weighted_imbalance(levels=5):
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

def rolling_ofi_sum():
    return sum(ofi_window)

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
# HMA + T3 CROSSING STRATEGY
# =========================
hma_values = deque(maxlen=100)
t3_values  = deque(maxlen=100)

def average_cross_signal(hma_vals, t3_vals):
    if len(hma_vals) < 2 or len(t3_vals) < 2:
        return "NEUTRAL ➖"
    hma_prev, hma_curr = hma_vals[-2], hma_vals[-1]
    t3_prev, t3_curr = t3_vals[-2], t3_vals[-1]
    if hma_prev < t3_prev and hma_curr > t3_curr:
        return "BUY 🟢"
    elif hma_prev > t3_prev and hma_curr < t3_curr:
        return "SELL 🔴"
    return "NEUTRAL ➖"

## =========================
# AUTO-CLOSE THRESHOLDS
# =========================
stop_loss_threshold = -0.1    # PnL threshold for stop-loss
take_profit_threshold = 0.2   # PnL threshold for take-profit

# =========================
# =========================
# FIBONACCI TREND MODULE
# =========================
def fibonacci_levels(prices, lookback=30):
    """
    Calculate Fibonacci retracement levels from recent high/low.
    """
    if len(prices) < lookback:
        return None
    recent_prices = prices[-lookback:]
    high = max(recent_prices)
    low = min(recent_prices)
    diff = high - low
    levels = {
        "0.0": high,
        "0.236": high - 0.236 * diff,
        "0.382": high - 0.382 * diff,
        "0.5": high - 0.5 * diff,
        "0.618": high - 0.618 * diff,
        "0.786": high - 0.786 * diff,
        "1.0": low
    }
    return levels

def fibonacci_signal(midprice, levels):
    """
    Generate BUY/SELL signal based on Fibonacci levels.
    - BUY near support levels (0.618, 0.786)
    - SELL near resistance levels (0.236, 0.382)
    """
    if levels is None: return "NEUTRAL ➖"
    support_levels = [levels["0.618"], levels["0.786"]]
    resistance_levels = [levels["0.236"], levels["0.382"]]

    # Close to support -> buy
    if any(abs(midprice - s)/s < 0.002 for s in support_levels):
        return "BUY 🟢"
    # Close to resistance -> sell
    elif any(abs(midprice - r)/r < 0.002 for r in resistance_levels):
        return "SELL 🔴"
    # Above 0.236 -> bullish
    elif midprice > levels["0.236"]:
        return "BULLISH 📈"
    # Below 0.786 -> bearish
    elif midprice < levels["0.786"]:
        return "BEARISH 📉"
    else:
        return "NEUTRAL ➖"


# =========================
# LIVE STREAM LOOP WITH FIBONACCI
# =========================
async def depth_stream():
    global balance, positions, trade_log
    print("🔵 Complete High-Frequency Multi-Strategy Engine with Fibonacci 📊\n")

    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # --- update orderbook
            for price, qty in msg["b"]:
                p, q = float(price), float(qty)
                if q == 0: bids.pop(p, None)
                else: bids[p] = q
            for price, qty in msg["a"]:
                p, q = float(price), float(qty)
                if q == 0: asks.pop(p, None)
                else: asks[p] = q
            if not bids or not asks: continue

            best_bid, best_ask = max(bids.keys()), min(asks.keys())
            midprice = (best_bid + best_ask) / 2

            # --- update price history
            price_history.append(midprice)
            if len(price_history) > window_size:
                price_history[:] = price_history[-window_size:]

            # --- trend model predictions
            preds = [
                predict_lr(price_history),
                predict_hma(price_history),
                predict_kalman(price_history),
                predict_cwma(price_history),
                predict_dma(price_history),
                predict_ema(price_history),
                predict_tema(price_history),
                predict_wma(price_history),
                predict_smma(price_history),
                predict_momentum(price_history),
                predict_hzlog(price_history),
                predict_vydia(price_history),
                predict_parma(price_history),
                predict_junx(price_history),
                predict_t3(price_history),
                predict_ichimoku(price_history)
            ]
            trend_final = merge_signals(preds, midprice)
            signals_dict = {f"M{i}": trend_signal(p, midprice) for i, p in enumerate(preds, 1)}

            # --- HFT & FPGA features (same as before)
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
                "price_skew": (p_skew, trend_signal(p_skew, 0))
            }

            # --- River online prediction
            next_pred, next_trend = update_river_models(midprice, {k: v[0] for k, v in fpga_features.items()})

            # --- Cache snapshot
            snapshot_cache.append({
                "midprice": midprice,
                **hft_features,
                **{k: v[0] for k, v in fpga_features.items()}
            })

            # --- HMA + T3 crossing
            hma_val = predict_hma(price_history)
            t3_val = predict_t3(price_history)
            hma_values.append(hma_val)
            t3_values.append(t3_val)
            cross_signal = average_cross_signal(hma_values, t3_values)

            # --- Fibonacci trend
            fib_levels = fibonacci_levels(price_history, lookback=30)
            fib_signal = fibonacci_signal(midprice, fib_levels)

            # =========================
            # MULTI-STRATEGY TRADING LOGIC
            # =========================
            # Trend Models M1-M16
            for name, signal in signals_dict.items():
                if signal == "BULLISH 📈":
                    if positions.get(name, {}).get("type") != "LONG":
                        if positions.get(name, {}).get("type") == "SHORT":
                            pnl = positions[name]["entry"] - midprice
                            balance += pnl
                            trade_log.append({"strategy": name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[name] = {"type": "LONG", "entry": midprice}
                        trade_log.append({"strategy": name, "side": "OPEN LONG", "price": midprice, "balance": balance})
                elif signal == "BEARISH 📉":
                    if positions.get(name, {}).get("type") != "SHORT":
                        if positions.get(name, {}).get("type") == "LONG":
                            pnl = midprice - positions[name]["entry"]
                            balance += pnl
                            trade_log.append({"strategy": name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                        positions[name] = {"type": "SHORT", "entry": midprice}
                        trade_log.append({"strategy": name, "side": "OPEN SHORT", "price": midprice, "balance": balance})

            # HMA+T3 crossing
            hma_name = "HMA+T3"
            if cross_signal == "BUY 🟢":
                if positions.get(hma_name, {}).get("type") != "LONG":
                    if positions.get(hma_name, {}).get("type") == "SHORT":
                        pnl = positions[hma_name]["entry"] - midprice
                        balance += pnl
                        trade_log.append({"strategy": hma_name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                    positions[hma_name] = {"type": "LONG", "entry": midprice}
                    trade_log.append({"strategy": hma_name, "side": "OPEN LONG", "price": midprice, "balance": balance})
            elif cross_signal == "SELL 🔴":
                if positions.get(hma_name, {}).get("type") != "SHORT":
                    if positions.get(hma_name, {}).get("type") == "LONG":
                        pnl = midprice - positions[hma_name]["entry"]
                        balance += pnl
                        trade_log.append({"strategy": hma_name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                    positions[hma_name] = {"type": "SHORT", "entry": midprice}
                    trade_log.append({"strategy": hma_name, "side": "OPEN SHORT", "price": midprice, "balance": balance})

            # Fibonacci strategy
            fib_name = "FIB"
            if fib_signal in ["BUY 🟢", "BULLISH 📈"]:
                if positions.get(fib_name, {}).get("type") != "LONG":
                    if positions.get(fib_name, {}).get("type") == "SHORT":
                        pnl = positions[fib_name]["entry"] - midprice
                        balance += pnl
                        trade_log.append({"strategy": fib_name, "side": "CLOSE SHORT", "price": midprice, "pnl": pnl, "balance": balance})
                    positions[fib_name] = {"type": "LONG", "entry": midprice}
                    trade_log.append({"strategy": fib_name, "side": "OPEN LONG", "price": midprice, "balance": balance})
            elif fib_signal in ["SELL 🔴", "BEARISH 📉"]:
                if positions.get(fib_name, {}).get("type") != "SHORT":
                    if positions.get(fib_name, {}).get("type") == "LONG":
                        pnl = midprice - positions[fib_name]["entry"]
                        balance += pnl
                        trade_log.append({"strategy": fib_name, "side": "CLOSE LONG", "price": midprice, "pnl": pnl, "balance": balance})
                    positions[fib_name] = {"type": "SHORT", "entry": midprice}
                    trade_log.append({"strategy": fib_name, "side": "OPEN SHORT", "price": midprice, "balance": balance})

            # =========================
            # AUTO-CLOSE POSITIONS
            # =========================
            for strat, pos in list(positions.items()):
                unrealized = (midprice - pos["entry"]) if pos["type"] == "LONG" else (pos["entry"] - midprice)
                if unrealized <= stop_loss_threshold or unrealized >= take_profit_threshold:
                    pnl = unrealized
                    balance += pnl
                    side = "CLOSE LONG" if pos["type"] == "LONG" else "CLOSE SHORT"
                    trade_log.append({"strategy": strat, "side": side, "price": midprice, "pnl": pnl, "balance": balance})
                    positions.pop(strat)

            # =========================
            # PRINT OUTPUT
            # =========================
            now = datetime.utcnow()
            print("\n⏱", now, "UTC")
            print("⭐ Trend Models Signals:", trend_final)
            for k, v in signals_dict.items(): print(f"   {k:7}: {v}")
            print("⭐ HMA + T3 Crossing Signal:", cross_signal)
            print("⭐ Fibonacci Signal:", fib_signal)
            print(f"⭐ Balance: {balance:.2f}")
            print("⭐ Current Positions:")
            for strat, pos in positions.items():
                unrealized = (midprice - pos["entry"]) if pos["type"] == "LONG" else (pos["entry"] - midprice)
                print(f"   {strat:10}: {pos['type']} @ {pos['entry']:.2f} | Unrealized PnL: {unrealized:.2f}")
            print("⭐ Last Trades:")
            for t in trade_log[-5:]:
                print(f"   {t['strategy']:10} {t['side']:15} @ {t['price']:.2f} | PnL: {t.get('pnl',0):.2f} | Balance: {t['balance']:.2f}")
            print("------------------------------------------------------------")

# =========================
# RUN
# =========================
await depth_stream()
