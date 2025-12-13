
import os
import asyncio
import aiohttp
import numpy as np
from collections import deque
from pykalman import KalmanFilter

# =========================
# EXCHANGES
# =========================
APIS = {
    "CoinGecko": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
    "Coinbase": "https://api.coinbase.com/v2/prices/spot?currency=USD",
    "Kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "Bitfinex": "https://api.bitfinex.com/v1/pubticker/btcusd",
    "Bitstamp": "https://www.bitstamp.net/api/v2/ticker/btcusd/"
}

# =========================
# UTILS
# =========================
def safe_return(val):
    return None if val is None or np.isnan(val) or np.isinf(val) else float(val)

def log_returns(prices):
    return np.diff(np.log(prices + 1e-8))

# =========================
# MODELS
# =========================
def predict_lr(prices):
    if len(prices) < 8:
        return None

    y = np.log(prices)
    x = np.arange(len(y))
    w = np.exp(np.linspace(-2, 0, len(y)))
    w /= w.sum()

    X = np.vstack([x, np.ones(len(x))]).T
    beta = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0]
    return safe_return(np.exp(beta[0] * len(y) + beta[1]))

def predict_hma(prices, period=16):
    if len(prices) < period:
        return None

    def wma(arr, n):
        w = np.arange(1, n + 1)
        return np.dot(arr[-n:], w) / w.sum()

    half = period // 2
    return safe_return(2 * wma(prices, half) - wma(prices, period))

def predict_kalman(prices):
    if len(prices) < 6:
        return None

    prices = np.log(prices)
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        transition_covariance=5e-5,
        observation_covariance=5e-4
    )
    state, _ = kf.filter(prices)
    return safe_return(np.exp(state[-1][0]))

def predict_cwma(prices):
    if len(prices) < 6:
        return None

    r = log_returns(np.array(prices))
    vol = np.std(r) + 1e-8
    w = np.exp(-r**2 / (2 * vol**2))
    w = np.insert(w, 0, w[0])
    w /= w.sum()
    return safe_return(np.sum(prices * w))

def predict_dma(prices, d=3):
    return safe_return(np.mean(prices[-d:]))

def predict_ema(prices, period=10):
    if len(prices) < period:
        return None

    r = log_returns(np.array(prices))
    vol = np.std(r)
    alpha = np.clip(1 / (1 + vol * 100), 0.1, 0.9)

    ema = prices[0]
    for p in prices[1:]:
        ema = alpha * p + (1 - alpha) * ema

    return safe_return(ema)

def predict_tema(prices, period=10):
    if len(prices) < period * 3:
        return None

    def ema(arr, p):
        a = 2 / (p + 1)
        e = arr[0]
        for x in arr[1:]:
            e = a * x + (1 - a) * e
        return e

    ema1 = ema(prices, period)
    ema2 = ema([ema1] * period, period)
    ema3 = ema([ema2] * period, period)

    return safe_return(3 * ema1 - 3 * ema2 + ema3)

def predict_wma(prices, p=10):
    if len(prices) < p:
        return None
    w = np.arange(1, p + 1)
    return safe_return(np.dot(prices[-p:], w) / w.sum())

def predict_smma(prices, p=10):
    if len(prices) < p:
        return None

    smma = np.mean(prices[:p])
    for x in prices[p:]:
        smma = (smma * (p - 1) + x) / p

    return safe_return(smma)

def predict_momentum(prices, p=5):
    if len(prices) < p + 2:
        return None

    prices = np.array(prices)
    r = log_returns(prices)
    return safe_return((prices[-1] - prices[-p - 1]) / (np.std(r[-p:]) + 1e-8))

MODELS = {
    "LR": predict_lr,
    "HMA": predict_hma,
    "Kalman": predict_kalman,
    "CWMA": predict_cwma,
    "DMA": predict_dma,
    "EMA": predict_ema,
    "TEMA": predict_tema,
    "WMA": predict_wma,
    "SMMA": predict_smma,
    "Momentum": predict_momentum
}

# =========================
# PAPER TRADER
# =========================
class PaperTrader:
    def __init__(self, balance=1000):
        self.balance = balance
        self.positions = {ex: {m: None for m in MODELS} for ex in APIS}
        self.pnl = {ex: {m: 0.0 for m in MODELS} for ex in APIS}

    def open_trade(self, ex, model, side, price):
        if self.positions[ex][model] is None:
            self.positions[ex][model] = {"side": side, "entry": price}

    def close_trade(self, ex, model, price):
        pos = self.positions[ex][model]
        if pos:
            pnl = (price - pos["entry"]) if pos["side"] == "buy" else (pos["entry"] - price)
            self.pnl[ex][model] += pnl
            self.balance += pnl
            self.positions[ex][model] = None

    def total_pnl(self):
        return sum(self.pnl[ex][m] for ex in APIS for m in MODELS)

# =========================
# FETCH PRICE
# =========================
async def fetch_price(ex, url, session):
    try:
        async with session.get(url, timeout=5) as r:
            d = await r.json()
            return ex, float(
                d["bitcoin"]["usd"] if ex == "CoinGecko" else
                d["data"]["amount"] if ex == "Coinbase" else
                d["last"] if ex == "Bitstamp" else
                d["last_price"] if ex == "Bitfinex" else
                list(d["result"].values())[0]["c"][0]
            )
    except:
        return ex, None

# =========================
# MAIN LOOP
# =========================
async def main():
    trader = PaperTrader()
    history = {ex: [] for ex in APIS}

    async with aiohttp.ClientSession() as session:
        while True:
            results = await asyncio.gather(*[
                fetch_price(ex, url, session) for ex, url in APIS.items()
            ])

            os.system("cls" if os.name == "nt" else "clear")
            print("🔵 BTC LIVE PRICE\n")

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
