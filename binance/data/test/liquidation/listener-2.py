
# ============================
# 🔥 SINGLE-CELL LIQUIDATION TRACKER + HEATMAP (COLAB SAFE)
# ============================

!pip install websockets seaborn --quiet

import nest_asyncio
nest_asyncio.apply()

import asyncio
import websockets
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import clear_output

# ----------------------
# Storage for events
# ----------------------
events = []
last_price = None

# ----------------------
# Timestamp helper
# ----------------------
def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

# ----------------------
# Event detection logic
# ----------------------
def detect_event(trade):
    global last_price, events

    price = float(trade["p"])
    qty   = float(trade["q"])
    side  = "BEARISH" if trade["m"] else "BULLISH"

    if last_price is None:
        last_price = price
        return

    price_diff = abs(price - last_price)

    # Liquidation heuristic:
    if qty >= 2 and price_diff >= 15:
        events.append({
            "timestamp": timestamp(),
            "side": side,
            "price": price,
            "volume": qty
        })
        print(f"🔥 Liquidation {timestamp()} | {side} | Price={price} | Vol={qty}")

    last_price = price

# ----------------------
# Render Heatmap
# ----------------------
def update_heatmap():
    if len(events) == 0:
        clear_output(wait=True)
        print("⏳ Waiting for liquidation events...")
        return

    df = pd.DataFrame(events)
    df["minute"] = df["timestamp"].str.slice(0, 16)

    heatmap_data = df.pivot_table(
        index="minute",
        columns="side",
        values="volume",
        aggfunc="sum",
        fill_value=0
    )

    clear_output(wait=True)
    plt.figure(figsize=(14, 6))
    sns.heatmap(
        heatmap_data,
        cmap="rocket_r",
        annot=True,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("🔥 BinanceUS Liquidation Heatmap (Bullish vs Bearish)")
    plt.xlabel("Side")
    plt.ylabel("Time (UTC Minute)")
    plt.xticks(rotation=0)
    plt.show()

    print(f"Total events collected: {len(events)}")

# ----------------------
# Main WebSocket Loop
# ----------------------
async def main():
    url = "wss://stream.binance.us:9443/ws/btcusdt@trade"
    print("📡 Connecting to BinanceUS WebSocket...")

    async with websockets.connect(url, ping_interval=20) as ws:
        print("✅ Connected! Collecting liquidation events...")
        counter = 0
        
        while True:
            msg = await ws.recv()
            trade = json.loads(msg)
            detect_event(trade)

            counter += 1
            if counter % 20 == 0:  # update heatmap every 20 trades
                update_heatmap()

# ----------------------
# Run inside cell (Colab-safe)
# ----------------------
await main()
