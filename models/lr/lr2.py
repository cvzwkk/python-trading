import nest_asyncio
nest_asyncio.apply()  # allow nested asyncio loops in Colab/Jupyter

import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import json
from collections import deque
from datetime import datetime
from IPython.display import clear_output, display

# =========================
# EXCHANGES (ORDERBOOK)
# =========================
ORDERBOOK_APIS = {
    "Coinbase": "https://api.exchange.coinbase.com/products/BTC-USD/book?level=2",
    "Kraken": "https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=10",
    "Bitstamp": "https://www.bitstamp.net/api/v2/order_book/btcusd/",
    "Bitfinex": "https://api.bitfinex.com/v1/book/btcusd"
}

BALANCE_FILE = "balance.json"

# =========================
# UTILS
# =========================
def safe_return(v):
    return None if v is None or np.isnan(v) or np.isinf(v) else float(v)

def log_returns(prices):
    return np.diff(np.log(prices + 1e-8))

def micro_price(bid, ask, bid_sz, ask_sz):
    return (ask * bid_sz + bid * ask_sz) / (bid_sz + ask_sz + 1e-8)

def save_trader_state(trader):
    state = {
        "balance": trader.balance,
        "pnl": trader.pnl,
        "positions": trader.positions
    }
    with open(BALANCE_FILE, "w") as f:
        json.dump(state, f)

def load_trader_state():
    if os.path.exists(BALANCE_FILE):
        with open(BALANCE_FILE, "r") as f:
            return json.load(f)
    return None

# =========================
# MODELS
# =========================
def predict_lr(prices):
    """Predict next 1-minute price from last 60s of prices"""
    n = len(prices)
    if n < 12:
        return None
    prices = np.array(prices, dtype=np.float64)
    logp = np.log(prices + 1e-9)
    r = np.diff(logp)
    alpha = 1.35
    fr = np.sign(r) * np.abs(r) ** alpha
    p = np.abs(fr) + 1e-12
    p /= p.sum()
    entropy = -np.sum(p * np.log(p))
    entropy_norm = entropy / np.log(len(p))
    entropy_weight = np.exp(-2.5 * entropy_norm)
    decay = np.exp(np.linspace(-4, 0, len(fr)))
    decay /= decay.sum()
    w = decay * entropy_weight
    w /= w.sum()
    velocity = np.sum(w * fr)
    acceleration = np.sum(w[:-1] * np.diff(fr))
    vol = np.std(r) + 1e-9
    vol_boost = np.tanh(vol * 80)
    drift = (velocity + 0.6 * acceleration) * (1 + vol_boost)
    z = (logp[-1] - logp.mean()) / (np.std(logp) + 1e-9)
    mr = np.tanh(-0.35 * z)
    final_drift = drift + mr * vol * 0.3
    horizon = 60  # 1-minute prediction
    forecast_log = logp[-1] + final_drift * horizon / max(1, n)
    return safe_return(np.exp(forecast_log))

MODELS = {"LR": predict_lr}

# =========================
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: None for e in ORDERBOOK_APIS}
        self.pnl = {e: 0.0 for e in ORDERBOOK_APIS}

        # Load saved state
        state = load_trader_state()
        if state:
            self.balance = state["balance"]
            self.pnl = state["pnl"]
            self.positions = state["positions"]

    def open_trade(self, ex, side, price):
        if self.positions[ex] is None:
            self.positions[ex] = {"side": side, "entry": price}

    def close_trade(self, ex, price):
        p = self.positions[ex]
        if p:
            pnl = price - p["entry"] if p["side"] == "buy" else p["entry"] - price
            self.balance += pnl
            self.pnl[ex] += pnl
            self.positions[ex] = None

    def total_pnl(self):
        return sum(self.pnl.values())

# =========================
# FETCH ORDERBOOK PRICE
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
# MAIN LOOP
# =========================
async def main():
    trader = PaperTrader()
    history = {e: deque(maxlen=60) for e in ORDERBOOK_APIS}

    async with aiohttp.ClientSession() as session:
        while True:
            results = await asyncio.gather(*[
                fetch_price(e, u, session) for e, u in ORDERBOOK_APIS.items()
            ])

            data_rows = []
            for ex, price in results:
                if price:
                    history[ex].append(price)

                    # Predict next 1-minute price
                    pred = None
                    if len(history[ex]) >= 12:
                        pred = MODELS["LR"](list(history[ex]))

                    pos = trader.positions[ex]
                    status = pos["side"].upper() if pos else None

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
                                status = None
                            elif pos["side"] == "sell" and pred > price + threshold:
                                trader.close_trade(ex, price)
                                status = None

                    data_rows.append({
                        "Exchange": ex,
                        "Price": price,
                        "Prediction_1min": pred,
                        "Position": status,
                        "PnL": trader.pnl[ex]
                    })

            df = pd.DataFrame(data_rows)
            df["Balance"] = trader.balance
            df["Total PnL"] = trader.total_pnl()

            clear_output(wait=True)  # <-- clear previous cell output
            print(f"🔵 BTC MICRO-PRICE + 1-MIN PREDICTIONS ({datetime.now().strftime('%H:%M:%S')})\n")
            display(df)

            save_trader_state(trader)  # persist every second
            await asyncio.sleep(1)

# =========================
# RUN IN COLAB/JUPYTER
# =========================
await main()
