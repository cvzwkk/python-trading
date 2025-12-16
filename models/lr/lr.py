
import os
import asyncio
import aiohttp
import numpy as np
from collections import deque
from pykalman import KalmanFilter

# =========================
# EXCHANGES (ORDERBOOK)
# =========================
ORDERBOOK_APIS = {
    "Coinbase": "https://api.exchange.coinbase.com/products/BTC-USD/book?level=2",
    "Kraken": "https://api.kraken.com/0/public/Depth?pair=XBTUSD&count=10",
    "Bitstamp": "https://www.bitstamp.net/api/v2/order_book/btcusd/",
    "Bitfinex": "https://api.bitfinex.com/v1/book/btcusd"
}

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
# MODELS
# =========================
def predict_lr(prices):
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

    horizon = min(6, n // 3)
    forecast_log = logp[-1] + final_drift * horizon

    return safe_return(np.exp(forecast_log))

MODELS = {
    "LR": predict_lr
}

# =========================
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: None for e in ORDERBOOK_APIS}
        self.pnl = {e: 0.0 for e in ORDERBOOK_APIS}

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

            else:
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
    history = {e: [] for e in ORDERBOOK_APIS}

    async with aiohttp.ClientSession() as session:
        while True:
            results = await asyncio.gather(*[
                fetch_price(e, u, session) for e, u in ORDERBOOK_APIS.items()
            ])

            os.system("cls" if os.name == "nt" else "clear")
            print("🔵 BTC MICRO-PRICE (ORDERBOOK)\n")

            for ex, price in results:
                if price:
                    history[ex].append(price)
                    history[ex] = history[ex][-60:]
                    print(f"{ex:10}: ${price:,.2f}")

            for ex, price in results:
                if not price or len(history[ex]) < 15:
                    continue

                vol = np.std(log_returns(np.array(history[ex]))) + 1e-8
                threshold = price * vol * (0.10 + 0.25 * np.tanh(vol * 50))

                for model, fn in MODELS.items():
                    pred = fn(history[ex])
                    if pred is None:
                        continue

                    pos = trader.positions[ex]

                    if pos is None:
                        if pred > price + threshold:
                            trader.open_trade(ex, "buy", price)
                        elif pred < price - threshold:
                            trader.open_trade(ex, "sell", price)
                    else:
                        if pos["side"] == "buy" and pred < price - threshold:
                            trader.close_trade(ex, price)
                        elif pos["side"] == "sell" and pred > price + threshold:
                            trader.close_trade(ex, price)

            print("\n📊 PnL PER EXCHANGE")
            for ex in ORDERBOOK_APIS:
                pos = trader.positions[ex]
                status = f" | OPEN {pos['side'].upper()}" if pos else ""
                print(f"{ex:10}: ${trader.pnl[ex]:>8.2f}{status}")

            print(
                f"\n💰 Balance: ${trader.balance:,.2f} | "
                f"Total PnL: ${trader.total_pnl():,.2f}"
            )

            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
