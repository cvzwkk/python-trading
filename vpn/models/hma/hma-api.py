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
from pykalman import KalmanFilter

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
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: None for e in ORDERBOOK_APIS}
        self.pnl = {e: 0.0 for e in ORDERBOOK_APIS}
        self.trade_history = deque(maxlen=50)

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
# MODELS
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
    return safe_return(np.exp(s[-1][0]))

def predict_ichimoku(prices):
    if len(prices) < 26:
        return None
    high = np.array(prices)
    low = np.array(prices)
    conv_line = (high[-9:].max() + low[-9:].min()) / 2
    base_line = (high[-26:].max() + low[-26:].min()) / 2
    span_a = (conv_line + base_line) / 2
    span_b = (high[-52:].max() + low[-52:].min()) / 2
    return safe_return((span_a + span_b) / 2)

def predict_junx(prices):
    return safe_return(np.mean(prices[-5:]))

def predict_parma(prices):
    ma = np.mean(prices[-10:])
    vol = np.std(prices[-10:])
    return safe_return(ma + vol*0.1)

def predict_hzlog(prices):
    x = np.arange(len(prices))
    y = np.log(prices + 1e-9)
    coef = np.polyfit(x, y, 1)
    return safe_return(np.exp(coef[1] + coef[0]*len(prices)))

def predict_madrid(prices):
    weights = np.linspace(1, 2, len(prices))
    return safe_return(np.dot(prices, weights)/weights.sum())

def predict_ribbon(prices):
    emas = [pd.Series(prices).ewm(span=s).mean().iloc[-1] for s in [5, 10, 20, 30]]
    return safe_return(np.mean(emas))

MODELS = {
    "Kalman": predict_kalman,
    "Ichimoku": predict_ichimoku,
    "Junx": predict_junx,
    "Parma": predict_parma,
    "HzLog": predict_hzlog,
    "Madrid": predict_madrid,
    "Ribbon": predict_ribbon
}

# =========================
# MODEL CONFIGURATION
# =========================
MODEL_MIN_HISTORY = {
    "Kalman": 6,
    "Ichimoku": 26,
    "Junx": 5,
    "Parma": 10,
    "HzLog": 5,
    "Madrid": 5,
    "Ribbon": 30
}

EXCHANGE_MODELS = {
    "Coinbase": ["Kalman", "Ichimoku"],
    "Kraken": ["Ichimoku", "Ribbon"],
    "Bitstamp": ["Kalman", "Parma"],
    "Bitfinex": ["Ribbon", "Junx"]
}

# =========================
# GLOBAL STATE
# =========================
history = {e: deque(maxlen=60) for e in ORDERBOOK_APIS}
trader = PaperTrader()
latest_results = {}

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
# BACKGROUND PRICE UPDATES
# =========================
async def update_prices():
    global latest_results
    async with aiohttp.ClientSession() as session:
        while True:
            results = await asyncio.gather(*[
                fetch_price(ex, url, session) for ex, url in ORDERBOOK_APIS.items()
            ])

            for ex, price in results:
                if price is None:
                    continue

                # Append latest price
                history[ex].append(price)
                latest_price = history[ex][-1]

                # Ensemble prediction
                ensemble_models = EXCHANGE_MODELS.get(ex, ["Kalman"])
                model_preds = []
                models_used = []
                last_prices_used = {}

                for model_name in ensemble_models:
                    min_len = MODEL_MIN_HISTORY.get(model_name, 6)
                    recent_prices = list(history[ex])[-min_len:]
                    if len(recent_prices) >= min_len:
                        pred = MODELS[model_name](recent_prices)
                        if pred is not None:
                            model_preds.append(pred)
                            models_used.append(model_name)
                            last_prices_used[model_name] = recent_prices

                final_pred = float(np.mean(model_preds)) if model_preds else None

                # Trade logic
                pos = trader.positions[ex]
                status = pos["side"].upper() if pos else "-"
                if final_pred is not None:
                    vol = np.std(log_returns(np.array(list(history[ex])[-max(MODEL_MIN_HISTORY.values()):]))) + 1e-8
                    threshold = latest_price * vol * 0.2

                    if pos is None:
                        if final_pred > latest_price + threshold:
                            trader.open_trade(ex, "buy", latest_price)
                            status = "BUY"
                        elif final_pred < latest_price - threshold:
                            trader.open_trade(ex, "sell", latest_price)
                            status = "SELL"
                    else:
                        if pos["side"] == "buy" and final_pred < latest_price - threshold:
                            trader.close_trade(ex, latest_price)
                            status = "-"
                        elif pos["side"] == "sell" and final_pred > latest_price + threshold:
                            trader.close_trade(ex, latest_price)
                            status = "-"

                latest_results[ex] = {
                    "price": latest_price,
                    "prediction": final_pred,
                    "models_used": models_used,
                    "position": status,
                    "pnl": trader.pnl[ex],
                    "last_prices_used": last_prices_used
                }

            await asyncio.sleep(1)

# =========================
# FASTAPI APP
# =========================
app = FastAPI(title="BTC Live Microprice API")

@app.get("/live")
async def live_data():
    trades = list(trader.trade_history)
    exchanges_info = {}

    for ex in ORDERBOOK_APIS:
        info = latest_results.get(ex, {})
        exchanges_info[ex] = {
            "latest_price": info.get("price"),
            "prediction": info.get("prediction"),
            "models_used": info.get("models_used", []),
            "position": info.get("position"),
            "pnl": info.get("pnl"),
            "last_prices_used": info.get("last_prices_used", {})
        }

    return JSONResponse({
        "timestamp": datetime.now().isoformat(),
        "balance": trader.balance,
        "total_pnl": trader.total_pnl(),
        "exchanges": exchanges_info,
        "last_trades": trades
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
