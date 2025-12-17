#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# IMPORTS
# =========================
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pyngrok import ngrok
import uvicorn

# =========================
# NGROK AUTH (OPTIONAL)
# =========================
NGROK_AUTHTOKEN = "36xhpiAn5cRi9ObeqeKYdJBZ13k_3z1GytiAf4Sn3czxWwNBm"
ngrok.set_auth_token(NGROK_AUTHTOKEN)

# =========================
# EXCHANGES & UTILS
# =========================
ORDERBOOK_APIS = {
    "Coinbase": "https://api.exchange.coinbase.com/products/BTC-USD/book?level=2",
    "Kraken": "https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=10",
    "Bitstamp": "https://www.bitstamp.net/api/v2/order_book/btcusd/",
    "Bitfinex": "https://api.bitfinex.com/v1/book/btcusd"
}

def safe_return(v):
    return None if v is None or np.isnan(v) or np.isinf(v) else float(v)

def log_returns(prices):
    return np.diff(np.log(prices + 1e-8))

def micro_price(bid, ask, bid_sz, ask_sz):
    return (ask * bid_sz + bid * ask_sz) / (bid_sz + ask_sz + 1e-8)

# =========================
# HMA MODEL
# =========================
def predict_hma_robust(prices, period=16):
    if len(prices) < 4:
        return None
    prices = np.array(prices, dtype=np.float64)
    prices = pd.Series(prices).ffill().bfill().values

    def wma(arr, n):
        n = min(len(arr), n)
        weights = np.arange(1, n + 1)
        return np.dot(arr[-n:], weights) / weights.sum()

    half = max(2, period // 2)
    half = min(half, len(prices))
    period = min(period, len(prices))

    hma = 2 * wma(prices, half) - wma(prices, period)

    slope_len = min(half, len(prices)-1)
    slope = np.polyfit(np.arange(slope_len+1), prices[-slope_len-1:], 1)[0]

    returns = np.diff(np.log(prices + 1e-9))
    momentum = np.sum(np.exp(-np.linspace(0,3,len(returns))) * returns) if len(returns) > 1 else 0.0
    vol = np.std(returns[-half:]) + 1e-9
    vol_boost = np.tanh(vol * 80)
    log_prices = np.log(prices + 1e-9)
    z = (log_prices[-1] - log_prices.mean()) / (np.std(log_prices) + 1e-9)
    mr_factor = np.tanh(-0.3 * z)
    forecast = hma + slope * (1 + vol_boost) + momentum * 0.5 + mr_factor * vol * 0.3
    return safe_return(forecast)

MODELS = {"HMA": predict_hma_robust}

# =========================
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: None for e in ORDERBOOK_APIS}
        self.pnl = {e: 0.0 for e in ORDERBOOK_APIS}

    def open_trade(self, ex, side, price, size=1.0):
        if self.positions[ex] is None:
            self.positions[ex] = {"side": side, "entry": price, "size": size}

    def close_trade(self, ex, price):
        p = self.positions[ex]
        if p:
            size = p.get("size", 1.0)
            pnl = (price - p["entry"]) * size if p["side"] == "buy" else (p["entry"] - price) * size
            self.balance += pnl
            self.pnl[ex] += pnl
            self.positions[ex] = None

    def total_pnl(self):
        return sum(self.pnl.values())

# =========================
# FETCH ORDERBOOK
# =========================
async def fetch_price(ex, url, session):
    try:
        async with session.get(url, timeout=5) as r:
            d = await r.json()
            if ex == "Coinbase":
                bid, bid_sz = map(float, d["bids"][0])
                ask, ask_sz = map(float, d["asks"][0])
            elif ex == "Kraken":
                book = list(d["result"].values())[0]
                bid, bid_sz = map(float, book["bids"][0][:2])
                ask, ask_sz = map(float, book["asks"][0][:2])
            elif ex == "Bitstamp":
                bid, bid_sz = float(d["bids"][0][0]), float(d["bids"][0][1])
                ask, ask_sz = float(d["asks"][0][0]), float(d["asks"][0][1])
            else:  # Bitfinex
                bid = float(d["bids"][0]["price"])
                bid_sz = float(d["bids"][0]["amount"])
                ask = float(d["asks"][0]["price"])
                ask_sz = float(d["asks"][0]["amount"])
            return ex, micro_price(bid, ask, bid_sz, ask_sz)
    except:
        return ex, None

# =========================
# GLOBAL STATE
# =========================
history = {e: deque(maxlen=60) for e in ORDERBOOK_APIS}
trader = PaperTrader()
latest_results = {}

# =========================
# BACKGROUND PRICE UPDATES
# =========================
async def update_prices():
    global latest_results
    async with aiohttp.ClientSession() as session:
        while True:
            results = await asyncio.gather(*[
                fetch_price(e, u, session) for e, u in ORDERBOOK_APIS.items()
            ])
            for ex, price in results:
                if price is not None:
                    history[ex].append(price)
                    pred = MODELS["HMA"](list(history[ex])) if len(history[ex]) >= 12 else None
                    pos = trader.positions[ex]
                    status = pos["side"].upper() if pos else "-"
                    if pred is not None:
                        vol = np.std(log_returns(np.array(history[ex]))) + 1e-8
                        threshold = price * vol * 0.2
                        if pos is None:
                            if pred > price + threshold:
                                trader.open_trade(ex, "buy", price)
                                status = "BUY"
                            elif pred < price - threshold:
                                trader.open_trade(ex, "sell", price)
                                status = "SELL"
                        else:
                            if pos["side"] == "buy" and pred < price - threshold:
                                trader.close_trade(ex, price)
                                status = "-"
                            elif pos["side"] == "sell" and pred > price + threshold:
                                trader.close_trade(ex, price)
                                status = "-"
                    latest_results[ex] = {
                        "price": price,
                        "prediction": pred,
                        "position": status,
                        "pnl": trader.pnl[ex]
                    }
            await asyncio.sleep(1)

# =========================
# FASTAPI
# =========================
app = FastAPI(title="BTC Live Microprice API")

@app.get("/live")
async def live_data():
    return JSONResponse({
        "timestamp": datetime.now().isoformat(),
        "balance": trader.balance,
        "total_pnl": trader.total_pnl(),
        "exchanges": latest_results
    })

# =========================
# MAIN ENTRY
# =========================
async def main():
    # start price updater
    asyncio.create_task(update_prices())

    # ngrok public URL
    public_url = ngrok.connect(8000, "http")
    print(f"🚀 Public URL: {public_url}")

    # start uvicorn
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
