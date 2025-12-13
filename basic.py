import os
import asyncio
import aiohttp
import numpy as np
from pykalman import KalmanFilter
from binance.client import Client

# =========================
# BINANCE.US CLIENT
# =========================
client = Client(tld="us")

# =========================
# EXCHANGES
# =========================
APIS = {
    "CoinGecko": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
    "Coinbase": "https://api.coinbase.com/v2/prices/spot?currency=USD",
    "Kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "Bitfinex": "https://api.bitfinex.com/v1/pubticker/btcusd",
    "Bitstamp": "https://www.bitstamp.net/api/v2/ticker/btcusd/",
    "Binance.US": "BINANCE"
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

        self.stats = {
            ex: {m: {"wins": 0, "losses": 0, "win_pnl": 0.0, "loss_pnl": 0.0}
                 for m in MODELS}
            for ex in APIS
        }

    def kelly_fraction(self, ex, model, cap=0.25):
        s = self.stats[ex][model]
        total = s["wins"] + s["losses"]
        if total < 10:
            return 0.05

        win_rate = s["wins"] / total
        avg_win = s["win_pnl"] / s["wins"] if s["wins"] else 0
        avg_loss = s["loss_pnl"] / s["losses"] if s["losses"] else 0

        if avg_loss == 0:
            return 0.05

        k = win_rate - (1 - win_rate) / (avg_win / avg_loss)
        return max(0.0, min(k * cap, cap))

    def calculate_position_size(self, ex, model, price, prices):
        if len(prices) < 2:
            return 0.001

        volatility = np.mean(np.abs(np.diff(prices[-10:])))
        if volatility <= 0:
            return 0.001

        risk_size = (self.balance * 0.002) / volatility
        kelly_size = (self.balance * self.kelly_fraction(ex, model)) / price

        size = min(risk_size, kelly_size)
        return float(np.clip(size, 0.001, 1.0))

    def open_trade(self, ex, model, side, price, size):
        if self.positions[ex][model] is None:
            self.positions[ex][model] = {
                "side": side,
                "entry": price,
                "size": size
            }

    def close_trade(self, ex, model, price):
        pos = self.positions[ex][model]
        if not pos:
            return

        pnl = ((price - pos["entry"]) if pos["side"] == "buy"
               else (pos["entry"] - price)) * pos["size"]

        self.balance += pnl
        self.pnl[ex][model] += pnl

        if pnl > 0:
            self.stats[ex][model]["wins"] += 1
            self.stats[ex][model]["win_pnl"] += pnl
        else:
            self.stats[ex][model]["losses"] += 1
            self.stats[ex][model]["loss_pnl"] += abs(pnl)

        self.positions[ex][model] = None

    def open_trades(self):
        return [(ex, m, p) for ex in APIS for m, p in self.positions[ex].items() if p]

    def total_pnl(self):
        return sum(self.pnl[ex][m] for ex in APIS for m in MODELS)

# =========================
# FETCH PRICE
# =========================
async def fetch_price(ex, url, session):
    try:
        if ex == "Binance.US":
            return ex, float(client.get_symbol_ticker(symbol="BTCUSDT")["price"])
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
                        if pred != price:
                            side = "buy" if pred > price else "sell"
                            size = trader.calculate_position_size(ex, model, price, history[ex])
                            trader.open_trade(ex, model, side, price, size)
                    else:
                        if (pos["side"] == "buy" and pred < price) or \
                           (pos["side"] == "sell" and pred > price):
                            trader.close_trade(ex, model, price)

            print("\n📊 OPEN TRADES")
            for ex, m, pos in trader.open_trades():
                print(f"{ex:10} | {m:9} | {pos['side'].upper():4} | {pos['entry']:.2f} | {pos['size']:.4f}")

            print(f"\n💰 Balance: ${trader.balance:,.2f} | PnL: ${trader.total_pnl():,.2f}")

            await asyncio.sleep(1)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    asyncio.run(main())
