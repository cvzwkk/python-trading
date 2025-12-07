
# =========================
# INSTALL DEPENDENCIES
# =========================
!pip install python-binance pykalman websockets nest_asyncio river --quiet

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

nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
window_size = 30        # trend model window
cache_window = 300      # last 5 minutes for River ML
price_history = []

# Simulation parameters
USD_balance = 10000.0
BTC_balance = 0.0
trade_size_pct = 0.5    # base size of trades
last_signal = None
trade_cache = []

# Tick-trading parameters
tick_threshold = 0.001  # 0.1% predicted price change
confidence_window = 20  # last predictions to measure confidence

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
pred_cache = deque(maxlen=confidence_window)

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
# LIVE STREAM LOOP WITH TICK TRADING
# =========================
async def depth_stream():
    global last_signal, USD_balance, BTC_balance, trade_cache
    print("🔵 HFT Engine with Tick Prediction & Risk-Weighted Trading\n")
    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg=json.loads(msg)
            for price, qty in msg["b"]:
                p=float(price); q=float(qty)
                if q==0: bids.pop(p,None)
                else: bids[p]=q
            for price, qty in msg["a"]:
                p=float(price); q=float(qty)
                if q==0: asks.pop(p,None)
                else: asks[p]=q
            if not bids or not asks: continue

            best_bid,max_best_ask = max(bids.keys()),min(asks.keys())
            midprice = (best_bid+max_best_ask)/2
            price_history.append(midprice)
            if len(price_history)>window_size: price_history[:] = price_history[-window_size:]

            # Predictions
            preds=[
                predict_lr(price_history),
                predict_hma(price_history),
                predict_kalman(price_history),
                predict_cwma(price_history),
                predict_dma(price_history),
                predict_ema(price_history),
                predict_tema(price_history),
                predict_wma(price_history),
                predict_smma(price_history)
            ]
            weights = np.array([1,1.2,1.1,0.9,1,1.3,1.3,1,1])
            trend_final = merge_signals(preds, midprice, weights=weights)

            # HFT & FPGA features
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
            next_pred, next_trend = update_river_models(midprice,{k:v[0] for k,v in fpga_features.items()})
            pred_cache.append(next_pred)

            # Risk weighting based on recent prediction trend
            if len(pred_cache) >= 2:
                conf = np.mean(np.sign(np.diff(list(pred_cache))))
                risk_weight = min(max(abs(conf),0.1),1.0)
            else:
                risk_weight = 0.5

            # ===== Tick-based buy/sell =====
            signal_alert = None
            if next_pred > midprice*(1+tick_threshold):
                usd_to_spend = USD_balance*trade_size_pct*risk_weight
                if usd_to_spend>0:
                    btc_to_buy = usd_to_spend/midprice
                    USD_balance -= usd_to_spend
                    BTC_balance += btc_to_buy
                    trade_cache.append({"type":"BUY","price":midprice,"btc":btc_to_buy,"usd":usd_to_spend,"time":datetime.utcnow()})
                    signal_alert = f"💰 BUY ALERT! Entry Price: {midprice:.2f} | BTC: {btc_to_buy:.6f} | Weight: {risk_weight:.2f}"
                    last_signal="BUY"
            elif next_pred < midprice*(1-tick_threshold):
                btc_to_sell = BTC_balance*trade_size_pct*risk_weight
                if btc_to_sell>0:
                    usd_gained = btc_to_sell*midprice
                    BTC_balance -= btc_to_sell
                    USD_balance += usd_gained
                    trade_cache.append({"type":"SELL","price":midprice,"btc":btc_to_sell,"usd":usd_gained,"time":datetime.utcnow()})
                    signal_alert = f"📉 SELL ALERT! Entry Price: {midprice:.2f} | BTC: {btc_to_sell:.6f} | Weight: {risk_weight:.2f}"
                    last_signal="SELL"

            total_equity = USD_balance + BTC_balance*midprice

            # Print output
            now=datetime.utcnow()
            print("\n⏱",now,"UTC")
            print("⭐ Trend Models:", trend_final)
            for i,name in enumerate(["LR","HMA","Kalman","CWMA","DMA","EMA","TEMA","WMA","SMMA"]):
                print(f"   {name:7}: {trend_signal(preds[i], midprice)}")
            print("⭐ HFT Indicators:")
            for k,v in hft_features.items(): print(f"   {k:18}: {v}")
            print("⭐ FPGA Features:")
            for k,(val,sig) in fpga_features.items(): print(f"   {k:18}: {val} | {sig}")
            print("⭐ River Tick Prediction:", f"{next_pred:.2f} | {next_trend}")
            if signal_alert: print(signal_alert)
            print(f"💹 Simulated Balance: USD {USD_balance:.2f} | BTC {BTC_balance:.6f} | Total Equity: {total_equity:.2f} USD")
            print("------------------------------------------------------------")

# =========================
# RUN
# =========================
await depth_stream()
