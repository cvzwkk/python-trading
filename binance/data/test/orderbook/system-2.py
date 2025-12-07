
#!pip install nest_asyncio websockets pandas --quiet

import asyncio
import websockets
import json
import pandas as pd
from IPython.display import display, clear_output
from collections import deque
import nest_asyncio

nest_asyncio.apply()

# -----------------------
# PARAMETERS
# -----------------------
symbol = "btcusdt"
WS_URL = f"wss://stream.binance.us:9443/ws/{symbol}@depth"
depth_rows = 5  # top 5 bids/asks
min_trade_size = 0.5  # BTC
max_trades = 10  # keep last 10 large trades

# -----------------------
# ORDERBOOK STORAGE
# -----------------------
bids, asks = {}, {}
large_trades = deque(maxlen=max_trades)

# -----------------------
# UTILITIES
# -----------------------
def get_top_levels():
    top_bids = sorted(bids.keys(), reverse=True)[:depth_rows]
    top_asks = sorted(asks.keys())[:depth_rows]

    data = []
    for i in range(depth_rows):
        bid_price = top_bids[i] if i < len(top_bids) else None
        bid_qty   = bids[bid_price] if bid_price else None
        ask_price = top_asks[i] if i < len(top_asks) else None
        ask_qty   = asks[ask_price] if ask_price else None
        data.append([bid_price, bid_qty, ask_price, ask_qty])
    df = pd.DataFrame(data, columns=["Bid Price","Bid Qty","Ask Price","Ask Qty"])
    return df

def get_large_trades_df():
    df = pd.DataFrame(list(large_trades), columns=["Price","Side","Qty"])
    return df

# -----------------------
# LIVE WEBSOCKET LOOP
# -----------------------
async def depth_stream():
    async with websockets.connect(WS_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)

            # Update bids
            for price, qty in msg["b"]:
                p = float(price)
                q = float(qty)
                if q == 0:
                    bids.pop(p, None)
                else:
                    bids[p] = q
                    # Record large buy trades
                    if q >= min_trade_size:
                        large_trades.append([p, "BUY", q])

            # Update asks
            for price, qty in msg["a"]:
                p = float(price)
                q = float(qty)
                if q == 0:
                    asks.pop(p, None)
                else:
                    asks[p] = q
                    # Record large sell trades
                    if q >= min_trade_size:
                        large_trades.append([p, "SELL", q])

            # Display tables
            clear_output(wait=True)
            print("🔹 Top Orderbook (5 levels)")
            display(get_top_levels())
            print("\n🔸 Large Trades (> 0.5 BTC, last 10)")
            display(get_large_trades_df())

# -----------------------
# RUN
# -----------------------
await depth_stream()
