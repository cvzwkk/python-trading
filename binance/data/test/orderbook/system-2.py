
# =========================
# INSTALL DEPENDENCIES
# =========================
#!pip install python-binance websockets nest_asyncio pandas --quiet

# =========================
# IMPORTS
# =========================
import asyncio
import websockets
import json
from collections import deque
import pandas as pd
from IPython.display import display, clear_output
import nest_asyncio
from binance.client import Client

nest_asyncio.apply()

# =========================
# PARAMETERS
# =========================
symbol = "BTCUSDT"
depth_rows = 5
min_trade_size = 0.5
max_trades = 10
max_liquidations = 10

# Binance client
client = Client(tld="us", api_key="", api_secret="")

# WebSockets
ws_symbol = symbol.lower()
WS_DEPTH_URL = f"wss://stream.binance.us:9443/ws/{ws_symbol}@depth"
WS_LIQ_URL   = f"wss://fstream.binance.com/ws/{ws_symbol}@forceOrder"

# =========================
# ORDERBOOK AND EVENTS
# =========================
bids, asks = {}, {}
large_trades = deque(maxlen=max_trades)
liquidations = deque(maxlen=max_liquidations)

# =========================
# DATAFRAMES UTILITY
# =========================
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
    return pd.DataFrame(data, columns=["Bid Price","Bid Qty","Ask Price","Ask Qty"])

def get_large_trades_df():
    return pd.DataFrame(list(large_trades), columns=["Price","Side","Qty"])

def get_liquidations_df():
    return pd.DataFrame(list(liquidations), columns=["Price","Side","Qty"])

# =========================
# DEPTH STREAM
# =========================
async def depth_stream():
    async with websockets.connect(WS_DEPTH_URL) as ws:
        async for msg in ws:
            msg=json.loads(msg)

            # Update bids
            for price, qty in msg["b"]:
                p=float(price); q=float(qty)
                if q == 0: bids.pop(p,None)
                else:
                    bids[p]=q
                    if q >= min_trade_size:
                        large_trades.append([p,"BUY",q])

            # Update asks
            for price, qty in msg["a"]:
                p=float(price); q=float(qty)
                if q == 0: asks.pop(p,None)
                else:
                    asks[p]=q
                    if q >= min_trade_size:
                        large_trades.append([p,"SELL",q])

# =========================
# LIQUIDATION STREAM
# =========================
async def liquidation_stream():
    async with websockets.connect(WS_LIQ_URL) as ws:
        async for msg in ws:
            msg = json.loads(msg)
            liq_price = float(msg['o']['p'])
            liq_qty   = float(msg['o']['q'])
            side      = "LONG" if msg['o']['S']=="SELL" else "SHORT"
            liquidations.append([liq_price, side, liq_qty])

# =========================
# DISPLAY LOOP
# =========================
async def display_loop():
    while True:
        clear_output(wait=True)
        print("🔹 Top Orderbook (5 levels)")
        display(get_top_levels())
        print("\n🔸 Large Trades (> 0.5 BTC, last 10)")
        display(get_large_trades_df())
        print("\n🔺 Liquidations (last 10)")
        display(get_liquidations_df())
        await asyncio.sleep(1)

# =========================
# RUN ALL
# =========================
await asyncio.gather(
    depth_stream(),
    liquidation_stream(),
    display_loop()
        )    for i in range(depth_rows):
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
