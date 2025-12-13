
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
# MODELS (UNCHANGED)
# =========================
def predict_lr(prices):
    if len(prices) < 8:
        return None
    y = np.log(prices)
    x = np.arange(len(y))
    w = np.exp(np.linspace(-2, 0, len(y)))
    w /= w.sum()
    X = np.vstack([x, np.ones(len(x))]).T
    b = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return safe_return(np.exp(b[0] * len(y) + b[1]))

def predict_hma(prices, p=16):
    if len(prices) < p:
        return None
    def wma(a, n):
        w = np.arange(1, n + 1)
        return np.dot(a[-n:], w) / w.sum()
    return safe_return(2 * wma(prices, p // 2) - wma(prices, p))

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

def predict_cwma(prices):
    if len(prices) < 6:
        return None
    r = log_returns(np.array(prices))
    v = np.std(r) + 1e-8
    w = np.exp(-r**2 / (2 * v**2))
    w = np.insert(w, 0, w[0])
    w /= w.sum()
    return safe_return(np.sum(prices * w))

def predict_dma(prices, d=3):
    return safe_return(np.mean(prices[-d:]))

def predict_ema(prices, p=10):
    if len(prices) < p:
        return None
    r = log_returns(np.array(prices))
    alpha = np.clip(1 / (1 + np.std(r) * 100), 0.1, 0.9)
    e = prices[0]
    for x in prices[1:]:
        e = alpha * x + (1 - alpha) * e
    return safe_return(e)

def predict_tema(prices, p=10):
    if len(prices) < p * 3:
        return None
    def ema(a, n):
        k = 2 / (n + 1)
        e = a[0]
        for x in a[1:]:
            e = k * x + (1 - k) * e
        return e
    e1 = ema(prices, p)
    e2 = ema([e1]*p, p)
    e3 = ema([e2]*p, p)
    return safe_return(3*e1 - 3*e2 + e3)

def predict_wma(prices, p=10):
    if len(prices) < p:
        return None
    w = np.arange(1, p+1)
    return safe_return(np.dot(prices[-p:], w) / w.sum())

def predict_smma(prices, p=10):
    if len(prices) < p:
        return None
    s = np.mean(prices[:p])
    for x in prices[p:]:
        s = (s * (p - 1) + x) / p
    return safe_return(s)

def predict_momentum(prices, p=5):
    if len(prices) < p + 2:
        return None
    r = log_returns(np.array(prices))
    return safe_return((prices[-1] - prices[-p-1]) / (np.std(r[-p:]) + 1e-8))

# =========================
# EXTRA MODELS (UNCHANGED)
# =========================
def predict_zlema(prices, p=10):
    if len(prices) < p * 2:
        return None
    lag = (p - 1) // 2
    adj = prices[-1] + (prices[-1] - prices[-lag - 1])
    a = 2 / (p + 1)
    z = prices[0]
    for x in prices[1:]:
        z = a * x + (1 - a) * z
    return safe_return(a * adj + (1 - a) * z)

def predict_covwma(prices, p=12):
    if len(prices) < p:
        return None
    a = np.array(prices[-p:])
    r = log_returns(a)
    cv = np.std(r) / (np.mean(a) + 1e-8)
    w = np.exp(-cv * np.arange(p)[::-1])
    w /= w.sum()
    return safe_return(np.sum(a * w))

def predict_t3(prices, p=10, v=0.7):
    if len(prices) < p * 3:
        return None
    def ema(a, n):
        k = 2 / (n + 1)
        e = a[0]
        for x in a[1:]:
            e = k * x + (1 - k) * e
        return e
    e1 = ema(prices, p)
    e2 = ema([e1]*p, p)
    e3 = ema([e2]*p, p)
    return safe_return(
        (-v**3)*e3 +
        (3*v**2 + 3*v**3)*e2 +
        (-6*v**2 - 3*v - 3*v**3)*e1 +
        (1 + 3*v + v**3 + 3*v**2)*prices[-1]
    )

def predict_ichimoku(prices):
    if len(prices) < 52:
        return None
    t = (np.max(prices[-9:]) + np.min(prices[-9:])) / 2
    k = (np.max(prices[-26:]) + np.min(prices[-26:])) / 2
    return safe_return((t + k) / 2)

def predict_parma(prices, p=12):
    if len(prices) < p + 3:
        return None
    r = log_returns(np.array(prices))
    phase = np.tanh(np.mean(r[-p:]) * 100)
    a = np.clip(0.1 + abs(phase), 0.1, 0.9)
    v = prices[0]
    for x in prices[1:]:
        v = a * x + (1 - a) * v
    return safe_return(v)

def predict_junx(prices):
    if len(prices) < 10:
        return None
    p = np.array(prices[-10:])
    t = np.arange(len(p))
    v = np.polyfit(t, p, 1)[0]
    a = np.polyfit(t, p, 2)[0]
    return safe_return(p[-1] + v + 0.5 * a)

MODELS = {
    "LR": predict_lr, "HMA": predict_hma, "Kalman": predict_kalman,
    "CWMA": predict_cwma, "DMA": predict_dma, "EMA": predict_ema,
    "TEMA": predict_tema, "WMA": predict_wma, "SMMA": predict_smma,
    "Momentum": predict_momentum,
    "ZLEMA": predict_zlema, "CoVWMA": predict_covwma, "T3": predict_t3,
    "Ichimoku": predict_ichimoku, "PARMA": predict_parma, "JUNX": predict_junx
}

# =========================
# PAPER TRADER (UNCHANGED)
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {e: {m: None for m in MODELS} for e in ORDERBOOK_APIS}
        self.pnl = {e: {m: 0.0 for m in MODELS} for e in ORDERBOOK_APIS}

    def open_trade(self, ex, model, side, price):
        if self.positions[ex][model] is None:
            self.positions[ex][model] = {"side": side, "entry": price}

    def close_trade(self, ex, model, price):
        p = self.positions[ex][model]
        if p:
            pnl = price - p["entry"] if p["side"] == "buy" else p["entry"] - price
            self.balance += pnl
            self.pnl[ex][model] += pnl
            self.positions[ex][model] = None

    def total_pnl(self):
        return sum(self.pnl[e][m] for e in self.pnl for m in self.pnl[e])

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
# MAIN LOOP (UNCHANGED)
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
                threshold = price * vol * 0.15

                for model, fn in MODELS.items():
                    pred = fn(history[ex])
                    if pred is None:
                        continue

                    pos = trader.positions[ex][model]

                    if pos is None:
                        if pred > price + threshold:
                            trader.open_trade(ex, model, "buy", price)
                        elif pred < price - threshold:
                            trader.open_trade(ex, model, "sell", price)
                    else:
                        if (pos["side"] == "buy" and pred < price - threshold) or \
                           (pos["side"] == "sell" and pred > price + threshold):
                            trader.close_trade(ex, model, price)

            print(f"\n💰 Balance: ${trader.balance:,.2f} | Total PnL: ${trader.total_pnl():,.2f}")
            await asyncio.sleep(1)

if __name__ == "__main__":
    asyncio.run(main())
