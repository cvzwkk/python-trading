!pip install python-binance
!pip install websockets
!pip install aiohttp
!pip install binance
!pip install nest_asyncio

import nest_asyncio
nest_asyncio.apply()

import asyncio
import websockets
import json
from datetime import datetime

#########################################
# Settings
#########################################
SYMBOL = "btcusdt"   # lowercase symbol required
VOLUME_THRESHOLD = 2.0   # BTC
PRICE_IMPACT = 15        # USD

last_price = None

#########################################
# Utility
#########################################
def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def print_liquidation(price, qty, side):
    print(
        f"\n{timestamp()} | 🔥 POSSIBLE LIQUIDATION\n"
        f"Side: {side}\n"
        f"Price: {price}\n"
        f"Volume: {qty} BTC\n"
        f"-----------------------------------"
    )

#########################################
# Liquidation-like detector
#########################################
def detect_event(trade):
    global last_price

    price = float(trade["p"])
    qty   = float(trade["q"])
    side  = "SELL" if trade["m"] else "BUY"

    if last_price is None:
        last_price = price
        return

    price_diff = abs(price - last_price)

    # LIQUIDATION-LIKE CONDITIONS
    if qty >= VOLUME_THRESHOLD and price_diff >= PRICE_IMPACT:
        print_liquidation(price, qty, side)

    last_price = price

#########################################
# WebSocket consumer
#########################################
async def main():
    url = f"wss://stream.binance.us:9443/ws/{SYMBOL}@trade"
    print(f"📡 Connected: {url}\nListening for liquidation-like events...\n")

    async with websockets.connect(url) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            detect_event(data)

#########################################
# Run safely in Google Colab
#########################################
await main()
