
import asyncio
import json
import numpy as np
from datetime import datetime
import websockets

bids = {}
asks = {}
last_best_bid = None
last_best_ask = None

symbol = "btcusdt"
WS_URL = f"wss://stream.binance.us:9443/ws/{symbol}@depth"


# ------------------------------
# HFT Indicators
# ------------------------------
def compute_hft_indicators():
    global last_best_bid, last_best_ask

    if not bids or not asks:
        return None

    best_bid = max(bids.keys())
    best_ask = min(asks.keys())

    # Microprice
    w = bids[best_bid] + asks[best_ask]
    microprice = (best_bid * asks[best_ask] + best_ask * bids[best_bid]) / w

    # OFI (Order Flow Imbalance)
    ofi = 0
    if last_best_bid is not None:
        ofi += best_bid - last_best_bid
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
