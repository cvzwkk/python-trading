import os
import asyncio
import aiohttp
import numpy as np
import time
from collections import deque
from pykalman import KalmanFilter

# =========================
# EXCHANGES (NO BINANCE)
# =========================
APIS = {
    "CoinGecko": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
    "Coinbase": "https://api.coinbase.com/v2/prices/spot?currency=USD",
    "Kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "Bitfinex": "https://api.bitfinex.com/v1/pubticker/btcusd",
    "Bitstamp": "https://www.bitstamp.net/api/v2/ticker/btcusd/"
}

# =========================
# TREND MODELS
# =========================
def predict_lr(prices):
    if len(prices) < 2: return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    m, c = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
    return float(m * len(prices) + c)

def predict_hma(prices, period=16):
    if len(prices) < period: return prices[-1]
    w = np.arange(1, period + 1)
    return float(np.dot(prices[-period:], w) / w.sum())

def predict_kalman(prices):
    if len(prices) < 2: return prices[-1]
    kf = KalmanFilter(initial_state_mean=prices[0], n_dim_obs=1)
    return float(kf.smooth(np.array(prices))[0][-1])

def predict_cwma(prices): return float(np.mean(prices))
def predict_dma(prices, d=3): return float(np.mean(prices[-d:]))

def predict_ema(prices, p=10):
    if len(prices) < p: return prices[-1]
    w = np.exp(np.linspace(-1, 0, p))
    w /= w.sum()
    return float(np.convolve(prices[-p:], w, mode="valid")[0])

def predict_tema(prices, p=10):
    if len(prices) < p * 3: return prices[-1]
    ema1 = predict_ema(prices, p)
    ema2 = predict_ema([ema1], p)
    ema3 = predict_ema([ema2], p)
    return float(3 * ema1 - 3 * ema2 + ema3)

def predict_wma(prices, p=10):
    if len(prices) < p: return prices[-1]
    w = np.arange(1, p + 1)
    return float(np.dot(prices[-p:], w) / w.sum())

def predict_smma(prices, p=10):
    if len(prices) < p: return prices[-1]
    smma = np.mean(prices[:p])
    for x in prices[p:]:
        smma = (smma * (p - 1) + x) / p
    return float(smma)

def predict_momentum(prices, p=5):
    if len(prices) < p + 1: return 0.0
    return float(prices[-1] - prices[-p - 1])

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

    def open_trades(self):
        return [(ex, m, pos) for ex in APIS for m, pos in self.positions[ex].items() if pos]

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

def clear():
    os.system("cls" if os.name == "nt" else "clear")

# =========================
# MAIN LOOP
# =========================
async def main():
    trader = PaperTrader()
    history = {ex: [] for ex in APIS}

    last_total_pnl = 0.0
    pnl_per_second = deque(maxlen=60)

    async with aiohttp.ClientSession() as session:
        while True:
            results = await asyncio.gather(*[
                fetch_price(ex, url, session) for ex, url in APIS.items()
            ])

            clear()
            print("🔵 BTC LIVE PRICE\n")

            for ex, price in results:
                if price:
                    history[ex].append(price)
                    history[ex] = history[ex][-50:]
                    print(f"{ex:10}: ${price:,.2f}")

            for ex, price in results:
                if not price or len(history[ex]) < 10:
                    continue

                for model, fn in MODELS.items():
                    pred = fn(history[ex])
                    pos = trader.positions[ex][model]

                    if pos is None:
                        trader.open_trade(ex, model, "buy" if pred > price else "sell", price)
                    else:
                        if (pos["side"] == "buy" and pred < price) or \
                           (pos["side"] == "sell" and pred > price):
                            trader.close_trade(ex, model, price)

            total_pnl = trader.total_pnl()
            pnl_sec = total_pnl - last_total_pnl
            pnl_per_second.append(pnl_sec)

            pnl_min = sum(pnl_per_second)
            avg_pnl_sec = pnl_min / len(pnl_per_second) if pnl_per_second else 0.0
            last_total_pnl = total_pnl

            print("\n📊 OPEN TRADES")
            for ex, m, pos in trader.open_trades():
                print(f"{ex:10} | {m:9} | {pos['side'].upper():4} | {pos['entry']:.2f}")

            print("\n💰 ACCOUNT")
            print(f"Balance        : ${trader.balance:,.2f}")
            print(f"Total PnL      : ${total_pnl:,.2f}")
            print(f"PnL / Second   : ${pnl_sec:,.4f}")
            print(f"PnL / Minute   : ${pnl_min:,.4f}")
            print(f"Avg PnL / Sec  : ${avg_pnl_sec:,.4f}")

            await asyncio.sleep(1)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    asyncio.run(main())
