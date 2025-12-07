
!pip install websockets nest_asyncio

import asyncio
import websockets
import json
import nest_asyncio
from collections import deque
import pandas as pd
from datetime import datetime

nest_asyncio.apply()

# ------------------------------------------
# Binance.US Depth WebSocket URL
# ------------------------------------------
WS_URL = "wss://stream.binance.us:9443/ws/btcusdt@depth"

# Store latest depth updates
depth_updates = deque(maxlen=500)

# ------------------------------------------
# Handle websocket messages
# ------------------------------------------
async def depth_listener():
    print("📡 Connected to Binance.US Depth Stream (BTC/USDT)")
    async with websockets.connect(WS_URL) as ws:
        while True:
            msg = await ws.recv()
            data = json.loads(msg)

            event_time = datetime.fromtimestamp(data["E"]/1000)

            # Bids + Asks
            bids = data["b"]
            asks = data["a"]

            depth_updates.append({
                "time": event_time,
                "bids": bids,
                "asks": asks
            })

            # Print live top-of-book
            if len(bids) > 0 and len(asks) > 0:
                best_bid = bids[0]
                best_ask = asks[0]

                print(
                    f"\n⏱ {event_time} "
                    f"\nBest Bid:  {best_bid[0]}  x {best_bid[1]}"
                    f"\nBest Ask:  {best_ask[0]}  x {best_ask[1]}"
                )

# ------------------------------------------
# Run forever in Colab
# ------------------------------------------
asyncio.get_event_loop().run_until_complete(depth_listener())
