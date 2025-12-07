
import asyncio
import json
import numpy as np
from datetime import datetime
import websockets
from collections import deque

# -------------------------------
# Internal Orderbook Memory
# -------------------------------
bids = {}
asks = {}
last_best_bid = None
last_best_ask = None

# For VPIN & volatility
vpin_window = deque(maxlen=50)
vol_window = deque(maxlen=100)

symbol = "btcusdt"
WS_URL = f"wss://stream.binance.us:9443/ws/{symbol}@depth"


# -------------------------------
# HFT INDICATOR FUNCTIONS
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
        return 0
    # approximate slope ∂Q/∂P
    return np.polyfit(prices, quantities, 1)[0]


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
    return np.std(np.diff(vol_window))


def liquidity_shock():
    spread = spread_indicator()
    return spread > 1.5 * np.mean([abs(x) for x in vol_window]) if len(vol_window) > 10 else None


# naive version: if more than 30 updates in <20ms
def quote_stuffing_detector(update_timestamps):
    if len(update_timestamps) < 2:
        return False
    if update_timestamps[-1] - update_timestamps[0] < 0.020 and len(update_timestamps) > 20:
        return True
    return False


def market_stress_index():
    spread = spread_indicator()
    vol = short_term_volatility(last_best_bid or 0)
    if vol is None:
        return None
    return spread * vol


# -------------------------------
# MAIN COMPUTATION ENGINE
# -------------------------------
async def depth_stream():
    print("🔵 Live HFT engine started (silent orderbook)...\n")

    update_times = deque(maxlen=40)

    async with websockets.connect(WS_URL) as ws:
        async for message in ws:
            msg = json.loads(message)

            now = datetime.utcnow()
            update_times.append(now.timestamp())

            # -------------- Update orderbook (silent) --------------
            for price, qty in msg["b"]:
                p = float(price)
                q = float(qty)
                if q == 0: bids.pop(p, None)
                else: bids[p] = q

            for price, qty in msg["a"]:
                p = float(price)
                q = float(qty)
                if q == 0: asks.pop(p, None)
                else: asks[p] = q

            if not bids or not asks:
                continue

            # ------------------ Compute all indicators ------------------
            best_bid = max(bids.keys())
            best_ask = min(asks.keys())
            midprice = (best_bid + best_ask) / 2

            hft = {
                "microprice": microprice_indicator(),
                "ofi": order_flow_imbalance(),
                "spread": spread_indicator(),
                "bid_pressure": pressure_indicator()[0],
                "ask_pressure": pressure_indicator()[1],
                "pressure_ratio": pressure_indicator()[2],
                "orderbook_slope": orderbook_slope(),
                "imbalance": inventory_imbalance(),
                "vpin": vpin_indicator(midprice),
                "volatility": short_term_volatility(midprice),
                "liquidity_shock": liquidity_shock(),
                "quote_stuffing": quote_stuffing_detector(update_times),
                "stress": market_stress_index()
            }

            # ------------------ Print Clean Output ------------------
            print("\n⏱", now, "UTC")
            for k, v in hft.items():
                print(f"{k:18}: {v}")
            print("--------------------------------------------------")


# -------------------------------
# COLAB-SAFE START
# -------------------------------
await depth_stream()        ofi += best_bid - last_best_bid
    if last_best_ask is not None:
        ofi += last_best_ask - best_ask

    last_best_bid = best_bid
    last_best_ask = best_ask

    # Bid/Ask pressure
    depth = 5
    bid_pressure = sum([bids[p] for p in sorted(bids.keys(), reverse=True)[:depth]])
    ask_pressure = sum([asks[p] for p in sorted(asks.keys())[:depth]])
    ratio = bid_pressure / ask_pressure if ask_pressure > 0 else None

    return {
        "microprice": microprice,
        "ofi": ofi,
        "bid_pressure": bid_pressure,
        "ask_pressure": ask_pressure,
        "ratio": ratio
    }


# ------------------------------
# WebSocket Depth Stream (Silent Mode)
# ------------------------------
async def depth_stream():
    print("🔵 Live HFT indicators started (depth is SILENT)...\n")

    async with websockets.connect(WS_URL) as ws:
        async for message in ws:
            msg = json.loads(message)

            # Update bids
            for price, qty in msg["b"]:
                p = float(price)
                q = float(qty)
                if q == 0:
                    bids.pop(p, None)
                else:
                    bids[p] = q

            # Update asks
            for price, qty in msg["a"]:
                p = float(price)
                q = float(qty)
                if q == 0:
                    asks.pop(p, None)
                else:
                    asks[p] = q

            # Compute only HFT indicators
            hft = compute_hft_indicators()
            if hft:
                print("\n⏱", datetime.utcnow(), "UTC")
                print("📌 Microprice:       ", hft["microprice"])
                print("📌 OFI:              ", hft["ofi"])
                print("📌 Bid Pressure:     ", hft["bid_pressure"])
                print("📌 Ask Pressure:     ", hft["ask_pressure"])
                print("📌 Pressure Ratio:   ", hft["ratio"])
                print("-----------------------------------------------")


# ------------------------------
# COLAB-SAFE START
# ------------------------------
await depth_stream()
