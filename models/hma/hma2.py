
import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import json
from collections import deque
from datetime import datetime

import nest_asyncio
nest_asyncio.apply()

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

# =========================
# LOAD / SAVE TRADER STATE
# =========================
def save_trader_state(trader):
    state = {
        "balance": trader.balance,
        "pnl": trader.pnl,
        "positions": trader.positions
    }
    if trader.balance is not None:
        with open(BALANCE_FILE, "w") as f:
            json.dump(state, f, indent=2)

def load_trader_state():
    if os.path.exists(BALANCE_FILE):
        try:
            state = json.load(open(BALANCE_FILE))
            # Ensure all positions have required keys
            for ex, pos in state.get("positions", {}).items():
                if pos is not None:
                    pos.setdefault("size", 1.0)
                    pos.setdefault("entry", pos.get("entry", 0.0))
            return state
        except json.JSONDecodeError:
            pass
    return {
        "balance": 1000.0,
        "pnl": {e: 0.0 for e in ORDERBOOK_APIS},
        "positions": {e: None for e in ORDERBOOK_APIS}
    }

# =========================
# MODELS
# =========================
def predict_hma_robust(prices, period=16):
    if len(prices) < 4:
        return None

    prices = np.array(prices, dtype=np.float64)
    prices = np.where(np.isfinite(prices), prices, np.nan)
    if np.isnan(prices).any():
        prices = pd.Series(prices).fillna(method="ffill").fillna(method="bfill").values

    def wma(arr, n):
        if len(arr) < n:
            n = len(arr)
        weights = np.arange(1, n + 1)
        return np.dot(arr[-n:], weights) / weights.sum()

    half = max(2, period // 2)
    hma = 2 * wma(prices, half) - wma(prices, period)
    recent_len = min(half, len(prices) - 1)
    slope = np.polyfit(np.arange(recent_len + 1), prices[-recent_len-1:], 1)[0]
    returns = np.diff(np.log(prices + 1e-9))
    momentum = np.sum(np.exp(-np.linspace(0,3,len(returns))) * returns) if len(returns) > 1 else 0.0
    vol = np.std(returns[-recent_len:]) + 1e-9
    vol_boost = np.tanh(vol * 80)
    log_prices = np.log(prices + 1e-9)
    z = (log_prices[-1] - log_prices.mean()) / (np.std(log_prices) + 1e-9)
    mr_factor = np.tanh(-0.3 * z)
    forecast = hma + slope * (1 + vol_boost) + momentum * 0.5 + mr_factor * vol * 0.3
    return safe_return(forecast)

def exotic_hma(prices, period=16):
    """Exotic, robust HMA + momentum + mean reversion + volatility adjustment."""
    try:
        prices = np.array(prices, dtype=np.float64)
        prices = np.where(np.isfinite(prices), prices, np.nan)
        prices = pd.Series(prices).fillna(method="ffill").fillna(method="bfill").values

        n = max(2, min(period, len(prices)))
        half = max(2, n // 2)
        
        # Weighted moving average
        def wma(arr, n):
            n = min(n, len(arr))
            weights = np.arange(1, n + 1)
            return np.dot(arr[-n:], weights) / weights.sum()

        hma = 2 * wma(prices, half) - wma(prices, n)

        # Slope / momentum
        x = np.arange(min(5, len(prices)))
        slope = np.polyfit(x, prices[-len(x):], 1)[0]

        # Volatility adjustment
        log_ret = np.diff(np.log(prices + 1e-8))
        vol = np.std(log_ret[-10:]) + 1e-8
        vol_boost = np.tanh(vol * 80)

        # Mean reversion factor
        z = (np.log(prices[-1]) - np.mean(np.log(prices))) / (np.std(np.log(prices)) + 1e-8)
        mr_factor = np.tanh(-0.3 * z)

        # Momentum factor
        momentum = np.sum(np.exp(-np.linspace(0,3,len(log_ret))) * log_ret) if len(log_ret) > 1 else 0.0

        forecast = hma + slope * (1 + vol_boost) + momentum * 0.5 + mr_factor * vol * 0.3
        return float(forecast)
    except:
        return float(prices[-1])  # fallback to last price if anything fails



MODELS = {"HMA": predict_hma_robust, "HMA2": exotic_hma}



# =========================
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: None for e in ORDERBOOK_APIS}
        self.pnl = {e: 0.0 for e in ORDERBOOK_APIS}

        state = load_trader_state()
        self.balance = state["balance"]
        self.pnl = state["pnl"]
        self.positions = state["positions"]
        # Ensure keys exist in positions
        for ex, pos in self.positions.items():
            if pos is not None:
                pos.setdefault("size", 1.0)
                pos.setdefault("entry", pos.get("entry", 0.0))

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

            os.system("cls" if os.name == "nt" else "clear")
            print(f"🔵 BTC MICRO-PRICE + 1-MIN PREDICTIONS ({datetime.now().strftime('%H:%M:%S')})\n")

            for ex, price in results:
                if price:
                    history[ex].append(price)
                    pred = MODELS["HMA"](list(history[ex])) if len(history[ex]) >= 12 else None
                    pos = trader.positions[ex]
                    status = pos["side"].upper() if pos else "-"

                    # Trading logic
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

                    # Real-time output
                    print(f"{ex:10} | Price: {price:10.2f} | Pred: {pred if pred else 'waiting':10} | "
                          f"Pos: {status:5} | PnL: {trader.pnl[ex]:10.2f}")

            print(f"\nBalance: {trader.balance:.2f} | Total PnL: {trader.total_pnl():.2f}")

            save_trader_state(trader)
            await asyncio.sleep(1)

# Run
await main()
