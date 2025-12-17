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
    # Only write file after trades affect balance/PnL
    if trader.balance is not None:
        with open(BALANCE_FILE, "w") as f:
            json.dump(state, f, indent=2)

def load_trader_state():
    if os.path.exists(BALANCE_FILE):
        try:
            with open(BALANCE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            pass
    # File missing or corrupted → start fresh without creating JSON
    return {
        "balance": 1000.0,
        "pnl": {e: 0.0 for e in ORDERBOOK_APIS},
        "positions": {e: None for e in ORDERBOOK_APIS}
    }

# =========================
# MODELS
# =========================
def predict_hma_robust(prices, period=16):
    """
    Maximum-performance Hull Moving Average predictor
    Returns next 1-min micro-price forecast
    """
    if len(prices) < 4:
        return None  # not enough data

    prices = np.array(prices, dtype=np.float64)

    # Handle NaNs or infs
    prices = np.where(np.isfinite(prices), prices, np.nan)
    if np.isnan(prices).any():
        prices = pd.Series(prices).fillna(method="ffill").fillna(method="bfill").values

    # Weighted Moving Average (WMA)
    def wma(arr, n):
        if len(arr) < n:
            n = len(arr)
        weights = np.arange(1, n + 1)
        return np.dot(arr[-n:], weights) / weights.sum()

    # Hull Moving Average (HMA)
    half = max(2, period // 2)
    hma = 2 * wma(prices, half) - wma(prices, period)

    # Trend velocity (recent slope)
    recent_len = min(half, len(prices) - 1)
    slope = np.polyfit(np.arange(recent_len + 1), prices[-recent_len-1:], 1)[0]

    # Momentum (weighted recent returns)
    returns = np.diff(np.log(prices + 1e-9))
    if len(returns) > 1:
        weights = np.exp(-np.linspace(0, 3, len(returns)))
        momentum = np.sum(weights * returns)
    else:
        momentum = 0.0

    # Volatility boost
    vol = np.std(returns[-recent_len:]) + 1e-9
    vol_boost = np.tanh(vol * 80)

    # Mean reversion factor
    log_prices = np.log(prices + 1e-9)
    z = (log_prices[-1] - log_prices.mean()) / (np.std(log_prices) + 1e-9)
    mr_factor = np.tanh(-0.3 * z)

    # Final adaptive forecast
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

        # Load saved state if exists
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
                        pred = MODELS["HMA"](list(history[ex]))

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

            os.system("cls" if os.name == "nt" else "clear")
            print(f"🔵 BTC MICRO-PRICE + 1-MIN PREDICTIONS ({datetime.now().strftime('%H:%M:%S')})\n")
            display(df)

            # Save trader state after any balance / PnL update
            save_trader_state(trader)

            await asyncio.sleep(1)

# Run
await main()
