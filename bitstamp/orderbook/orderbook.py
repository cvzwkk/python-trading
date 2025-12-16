
import asyncio
import json
import websockets
import nest_asyncio
from datetime import datetime
from IPython.display import clear_output

nest_asyncio.apply()

BITSTAMP_WS = "wss://ws.bitstamp.net"
SYMBOL = "btcusd"   # change if needed (ethusd, xrpusd, etc)
BOOK_LEVELS = 5

last_price = None
bids = []
asks = []

async def subscribe(ws, channel):
    payload = {
        "event": "bts:subscribe",
        "data": {"channel": channel}
    }
    await ws.send(json.dumps(payload))

async def stream():
    global last_price, bids, asks

    async with websockets.connect(BITSTAMP_WS) as ws:
        await subscribe(ws, f"live_trades_{SYMBOL}")
        await subscribe(ws, f"order_book_{SYMBOL}")

        while True:
            msg = json.loads(await ws.recv())

            if msg.get("event") != "data":
                continue

            channel = msg["channel"]
            data = msg["data"]

            if "live_trades" in channel:
                last_price = float(data["price"])

            elif "order_book" in channel:
                bids = [(float(p), float(q)) for p, q in data["bids"][:BOOK_LEVELS]]
                asks = [(float(p), float(q)) for p, q in data["asks"][:BOOK_LEVELS]]

            render()

def render():
    clear_output(wait=True)
    now = datetime.utcnow().strftime("%H:%M:%S")

    print("BITSTAMP LIVE MARKET (BTC/USD)")
    print(f"Time (UTC): {now}")
    print("-" * 45)

    if last_price:
        print(f"Last Price: {last_price:,.2f}\n")

    print("ASKS")
    print("Price        | Size")
    for p, q in reversed(asks):
        print(f"{p:12,.2f} | {q:,.6f}")

    print("\nBIDS")
    print("Price        | Size")
    for p, q in bids:
        print(f"{p:12,.2f} | {q:,.6f}")

# Run
asyncio.get_event_loop().run_until_complete(stream())
