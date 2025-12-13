#!pip install pykalman python-binance numpy
import os
import asyncio
import aiohttp
import numpy as np
from pykalman import KalmanFilter
from binance.client import Client

# =========================
# BINANCE.US CLIENT
# =========================
client = Client(tld='us')  # no API key required

# =========================
# EXCHANGE PRICE URLS
# =========================
APIS = {
    "CoinGecko": "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
    "Coinbase": "https://api.coinbase.com/v2/prices/spot?currency=USD",
    "Kraken": "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
    "Bitfinex": "https://api.bitfinex.com/v1/pubticker/btcusd",
    "Bitstamp": "https://www.bitstamp.net/api/v2/ticker/btcusd/",
    "Binance.US": "Binance.US Client"
}

# =========================
# PAPER TRADING STATE
# =========================
class PaperTrader:
    def __init__(self, initial_balance=1000):
        self.balance = initial_balance
        self.positions = {ex: None for ex in APIS.keys()}
        self.pnl = {ex: 0.0 for ex in APIS.keys()}

    def open_trade(self, exchange, trade_type, price, size=1):
        if self.positions[exchange] is None:
            self.positions[exchange] = {"type": trade_type, "price": price, "size": size}

    def close_trade(self, exchange, price):
        pos = self.positions[exchange]
        if pos:
            if pos["type"] == "buy":
                profit = (price - pos["price"]) * pos["size"]
            else:
                profit = (pos["price"] - price) * pos["size"]
            self.pnl[exchange] += profit
            self.balance += profit
            self.positions[exchange] = None

    def get_summary(self):
        summary = f"Balance: ${self.balance:,.2f} | Total PnL: ${sum(self.pnl.values()):,.2f}\n"
        for ex, pos in self.positions.items():
            if pos:
                summary += f"{ex:12}: OPEN {pos['type'].upper()} at ${pos['price']:.2f}\n"
            else:
                summary += f"{ex:12}: NO POSITION\n"
        return summary

    def get_open_trades(self):
        """Return only currently open trades"""
        return {ex: pos for ex, pos in self.positions.items() if pos is not None}

# =========================
# TREND PREDICTION MODELS
# =========================
def predict_lr(prices):
    if len(prices) < 2:
        return prices[-1]
    x = np.arange(len(prices))
    y = np.array(prices)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m * len(prices) + c)

# =========================
# FETCH PRICES
# =========================
async def fetch_price(name, url=None, session=None):
    try:
        if name == "Binance.US":
            price = client.get_symbol_ticker(symbol="BTCUSDT")
            return name, float(price['price'])
        else:
            async with session.get(url, timeout=5) as resp:
                data = await resp.json()
                if name == "CoinGecko":
                    return name, float(data["bitcoin"]["usd"])
                elif name == "Coinbase":
                    return name, float(data["data"]["amount"])
                elif name == "Kraken":
                    pair = list(data["result"].keys())[0]
                    return name, float(data["result"][pair]["c"][0])
                elif name == "Bitfinex":
                    return name, float(data["last_price"])
                elif name == "Bitstamp":
                    return name, float(data["last"])
        return name, None
    except Exception:
        return name, None

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

# =========================
# PAPER TRADING LOOP
# =========================
async def paper_trading_loop():
    trader = PaperTrader()
    prices_history = {ex: [] for ex in APIS.keys()}

    async with aiohttp.ClientSession() as session:
        while True:
            tasks = [fetch_price(name, url, session) for name, url in APIS.items()]
            results = await asyncio.gather(*tasks)

            clear_console()
            print("🔵 LIVE BTC PRICE (USD) 🔵\n")

            for name, price in results:
                if price is not None:
                    print(f"{name:12}: ${price:,.2f}")
                    prices_history[name].append(price)
                    if len(prices_history[name]) > 50:
                        prices_history[name].pop(0)
                else:
                    print(f"{name:12}: Error fetching price")

            # Apply trend model and manage trades
            for name, price in results:
                if price is None or len(prices_history[name]) < 5:
                    continue
                pos = trader.positions[name]
                pred = predict_lr(prices_history[name])

                if pos is None:
                    # No position, open new trade
                    if pred > price:
                        trader.open_trade(name, "buy", price)
                    elif pred < price:
                        trader.open_trade(name, "sell", price)
                else:
                    # Position exists, check if opposite signal to close
                    if pos["type"] == "buy" and pred < price:
                        trader.close_trade(name, price)
                        trader.open_trade(name, "sell", price)
                    elif pos["type"] == "sell" and pred > price:
                        trader.close_trade(name, price)
                        trader.open_trade(name, "buy", price)

            # =========================
            # PRINT OPEN TRADES
            # =========================
            print("\n📊 OPENED TRADES 📊")
            open_trades = trader.get_open_trades()
            if open_trades:
                for ex, pos in open_trades.items():
                    print(f"{ex:12}: {pos['type'].upper()} at ${pos['price']:.2f}")
            else:
                print("No open trades currently.")

            # Print balance & total PnL
            print(f"\nBalance: ${trader.balance:,.2f} | Total PnL: ${sum(trader.pnl.values()):,.2f}")

            await asyncio.sleep(1)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    asyncio.run(paper_trading_loop())
