#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pykalman import KalmanFilter
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import subprocess

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
# KALMAN MODEL
# =========================
def predict_kalman(prices):
    if len(prices) < 6:
        return None
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=5e-5,
        observation_covariance=5e-4
    )
    s, _ = kf.filter(np.log(prices))
    return safe_return(np.exp(s[-1][0])

MODELS = {"Kalman": predict_kalman}

# =========================
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: None for e in ORDERBOOK_APIS}
        self.pnl = {e: 0.0 for e in ORDERBOOK_APIS}
        self.trade_history = deque(maxlen=50)  # last 50 trades

    def open_trade(self, ex, side, price, size=1.0):
        if self.positions[ex] is None:
            self.positions[ex] = {"side": side, "entry": price, "size": size}
            self.trade_history.append({
                "exchange": ex,
                "type": "ENTRY",
                "side": side.upper(),
                "price": price,
                "pnl": None,
                "time": datetime.now().strftime("%H:%M:%S")
            })

    def close_trade(self, ex, price):
        p = self.positions[ex]
        if p:
            size = p.get("size", 1.0)
            pnl = (price - p["entry"]) * size if p["side"] == "buy" else (p["entry"] - price) * size
            self.balance += pnl
            self.pnl[ex] += pnl
            self.positions[ex] = None
            self.trade_history.append({
                "exchange": ex,
                "type": "EXIT",
                "side": p["side"].upper(),
                "price": price,
                "pnl": pnl,
                "time": datetime.now().strftime("%H:%M:%S")
            })

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
                    pred = MODELS["Kalman"](list(history[ex])) if len(history[ex]) >= 6 else None
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
# FASTAPI APP
# =========================
app = FastAPI(title="BTC Live Microprice API")

@app.get("/live")
async def live_data():
    trades = list(trader.trade_history)
    return JSONResponse({
        "timestamp": datetime.now().isoformat(),
        "balance": trader.balance,
        "total_pnl": trader.total_pnl(),
        "exchanges": latest_results,
        "last_trades": trades
    })

# =========================
# MAIN ENTRY
# =========================
async def main():
    # start price updater
    asyncio.create_task(update_prices())

    # Serveo SSH tunnel
    print("🌐 Starting Serveo tunnel...")
    subprocess.Popen(["ssh", "-o", "StrictHostKeyChecking=no", "-R", "80:localhost:8001", "serveo.net"])

    # start uvicorn
    config = uvicorn.Config(app=app, host="0.0.0.0", port=8001, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
