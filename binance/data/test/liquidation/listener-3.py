# Single Colab cell: Full HFT-like live collector + footprint + heatmaps + River online predictor
# Installs (silent)
!pip install websockets nest_asyncio python-binance river seaborn matplotlib --quiet

# ---- Imports ----
import nest_asyncio
nest_asyncio.apply()

import asyncio, websockets, json, time, math
from datetime import datetime, timezone
from collections import deque, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from binance.client import Client  # for orderbook snapshots
# River for online model
from river import preprocessing, linear_model, metrics

# ---- SETTINGS ----
SYMBOL = "btcusdt"             # websocket symbol (lowercase)
REST_SYMBOL = "BTCUSDT"        # REST symbol (uppercase) for orderbook
TRADE_VOLUME_THRESHOLD = 2.0   # BTC (heuristic large trade)
PRICE_IMPACT_THRESHOLD = 15.0  # USD (heuristic price jump)
BOOK_POLL_SECONDS = 5          # poll order book every N seconds (REST)
PRICE_BIN_SIZE = 50            # heatmap price bin size in USD
HEATMAP_TIME_WINDOW_MINUTES = 20  # number of minutes to show in heatmap
PLOT_UPDATE_EVERY = 20        # update plots every N trades received
POLL_ORDERBOOK_EVERY = 5      # seconds

# ---- Globals / Storage ----
events = []       # liquidation-like events: dict(timestamp, side, price, volume)
trades = deque(maxlen=5000)   # recent trades: dict(ts, price, qty, side)
price_buckets = defaultdict(lambda: {'buy':0.0, 'sell':0.0})  # price_bin -> volumes
imbalance_series = deque(maxlen=1000)  # (ts, imbalance)
timestamps_series = deque(maxlen=1000)
prediction_series = deque(maxlen=1000)  # (ts, prob)
orderbook_snapshots = deque(maxlen=200) # store recent orderbook snapshots

last_price = None
trade_counter = 0
last_book_poll = 0.0

# ---- Binance REST client for orderbook snapshots ----
rest_client = Client()  # no API key needed for public endpoints

# ---- River online model (classifier) ----
# We'll predict whether a trade is a "liquidation-like" event (1) or not (0)
model = preprocessing.StandardScaler() | linear_model.LogisticRegression()
metric_auc = metrics.ROCAUC()
metric_logloss = metrics.LogLoss()

# helper for time
def now_str():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

# ---- Utilities ----
def is_aggressive_sell(trade_msg):
    # 'm' field is maker: True if maker (i.e., trade was maker sell/buy). We treat 'm'=True as SELL taker? 
    # For simplicity: trade['m'] True => SELL taker (aggressive sell), else BUY taker
    return trade_msg.get('m', False)

def detect_liquidation_like(trade):
    """
    Heuristic: trade large volume + immediate price impact threshold.
    """
    global last_price
    price = float(trade["p"])
    qty = float(trade["q"])
    side = "BEARISH" if trade["m"] else "BULLISH"
    if last_price is None:
        last_price = price
        return False, None
    impact = abs(price - last_price)
    last_price = price
    if (qty >= TRADE_VOLUME_THRESHOLD) and (impact >= PRICE_IMPACT_THRESHOLD):
        ev = {
            "timestamp": now_str(),
            "side": side,
            "price": price,
            "volume": qty
        }
        events.append(ev)
        return True, ev
    return False, None

def update_price_buckets(price, qty, side):
    # bucket price to nearest PRICE_BIN_SIZE
    bin_floor = (math.floor(price / PRICE_BIN_SIZE) * PRICE_BIN_SIZE)
    key = int(bin_floor)
    if side.upper() == 'BUY' or side.upper()=='BULLISH':
        price_buckets[key]['buy'] += qty
    else:
        price_buckets[key]['sell'] += qty

def fetch_orderbook_snapshot():
    """Fetch orderbook (bids/asks) via REST and compute imbalance and top levels volumes."""
    try:
        ob = rest_client.get_order_book(symbol=REST_SYMBOL, limit=20)
        ts = now_str()
        bids = [(float(p), float(q)) for p,q in ob['bids']]
        asks = [(float(p), float(q)) for p,q in ob['asks']]
        sum_bids = sum([q for p,q in bids])
        sum_asks = sum([q for p,q in asks])
        imbalance = (sum_bids - sum_asks) / (sum_bids + sum_asks + 1e-9)
        snapshot = {
            "timestamp": ts,
            "bids": bids,
            "asks": asks,
            "sum_bids": sum_bids,
            "sum_asks": sum_asks,
            "imbalance": imbalance
        }
        orderbook_snapshots.append(snapshot)
        imbalance_series.append((ts, imbalance))
        timestamps_series.append(ts)
        return snapshot
    except Exception as e:
        print("Orderbook fetch error:", e)
        return None

# ---- Footprint helper: aggregate recent trades into price × time grid ----
def build_footprint_matrix(last_n_seconds=300, bin_size=PRICE_BIN_SIZE):
    """
    Creates a DataFrame where rows are time buckets (seconds or minutes) and columns are price bins.
    We'll use recent trades deque.
    """
    if len(trades) == 0:
        return None, None, None
    # convert trades to DataFrame
    df = pd.DataFrame(list(trades))
    # keep only last N seconds
    df['ts_dt'] = pd.to_datetime(df['ts'])
    cutoff = pd.Timestamp.utcnow(tz=timezone.utc) - pd.Timedelta(seconds=last_n_seconds)
    df = df[df['ts_dt'] >= cutoff]
    if df.empty:
        return None, None, None
    # time bucket to second
    df['time_bucket'] = df['ts_dt'].dt.strftime('%H:%M:%S')
    df['price_bin'] = (df['price'] // bin_size) * bin_size
    buys = df[df['side']=='BUY'].groupby(['time_bucket','price_bin'])['qty'].sum().unstack(fill_value=0)
    sells = df[df['side']=='SELL'].groupby(['time_bucket','price_bin'])['qty'].sum().unstack(fill_value=0)
    # footprint = buys - sells (positive means net buying)
    footprint = buys.reindex_like(sells).fillna(0) - sells.reindex_like(buys).fillna(0)
    # align columns sorted
    cols = sorted(set(df['price_bin']))
    footprint = footprint.reindex(columns=sorted(footprint.columns), fill_value=0)
    return footprint, buys, sells

# ---- Merge signals helper ----
def merged_signal_from_preds(price, preds):
    bullish = sum(1 for p in preds if p > price)
    bearish = sum(1 for p in preds if p < price)
    if bullish > bearish:
        return "BULLISH"
    elif bearish > bullish:
        return "BEARISH"
    else:
        return "NEUTRAL"

# ---- Plotting function ----
def update_plots():
    clear = True
    plt.clf()
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=(1,1,1.2))

    # PRICE + events (top-left)
    ax1 = fig.add_subplot(gs[0,0])
    if len(trades)>0:
        df_tr = pd.DataFrame(list(trades))
        df_tr['ts'] = pd.to_datetime(df_tr['ts'])
        # sample to last 300 points
        dfp = df_tr.tail(300)
        ax1.plot(dfp['ts'], dfp['price'], label='Price', color='black')
    # annotate events
    if len(events)>0:
        ev_df = pd.DataFrame(events)
        # convert timestamp for plotting
        try:
            ev_df['ts_dt'] = pd.to_datetime(ev_df['timestamp'])
        except:
            ev_df['ts_dt'] = pd.to_datetime(ev_df['timestamp'], utc=True, errors='coerce')
        for _, r in ev_df.tail(50).iterrows():
            ax1.scatter(r['ts_dt'], r['price'], s=80, marker='v' if r['side']=="BEARISH" else '^',
                        color='red' if r['side']=="BEARISH" else 'green')
    ax1.set_title("Price (recent) with liquidation-like spikes")
    ax1.tick_params(axis='x', rotation=30)

    # Heatmap: liquidation volume bullish vs bearish by minute (top-right)
    ax2 = fig.add_subplot(gs[0,1])
    if len(events)>0:
        evdf = pd.DataFrame(events)
        evdf['minute'] = evdf['timestamp'].str.slice(0,16)
        heat = evdf.pivot_table(index='minute', columns='side', values='volume', aggfunc='sum', fill_value=0)
        # ensure ordering latest at bottom
        heat = heat.tail(HEATMAP_TIME_WINDOW_MINUTES)
        sns.heatmap(heat.T, ax=ax2, cmap='rocket_r', annot=True, fmt=".2f")
        ax2.set_title("Liquidation Heatmap (minute × side)")
    else:
        ax2.text(0.5,0.5,"No events yet", ha='center')

    # Footprint (middle-left)
    ax3 = fig.add_subplot(gs[1,0])
    footprint, buys, sells = build_footprint_matrix(last_n_seconds=300, bin_size=PRICE_BIN_SIZE)
    if footprint is not None:
        sns.heatmap(footprint.replace(0, np.nan).fillna(0), ax=ax3, cmap='bwr', center=0)
        ax3.set_title("Footprint (buys - sells) recent seconds")
    else:
        ax3.text(0.5,0.5,"No trade footprint yet", ha='center')

    # Orderbook imbalance (middle-right)
    ax4 = fig.add_subplot(gs[1,1])
    if len(imbalance_series)>0:
        times = [t for t,_ in imbalance_series]
        vals = [v for _,v in imbalance_series]
        # convert to times for plotting
        ax4.plot(times[-100:], vals[-100:], marker='o')
        ax4.set_title("Orderbook Imbalance (recent)")
        ax4.tick_params(axis='x', rotation=45)
    else:
        ax4.text(0.5,0.5,"No orderbook data yet", ha='center')

    # Prediction probability (bottom-left)
    ax5 = fig.add_subplot(gs[2,0])
    if len(prediction_series)>0:
        times = [t for t,_ in prediction_series]
        probs = [p for _,p in prediction_series]
        ax5.plot(times, probs, label='Prob(liquidation)')
        ax5.set_ylim(0,1)
        ax5.set_title("River Online Model Probability")
        ax5.tick_params(axis='x', rotation=45)
    else:
        ax5.text(0.5,0.5,"No predictions yet", ha='center')

    # Volume clusters summary (bottom-right)
    ax6 = fig.add_subplot(gs[2,1])
    if len(events)>0:
        edf = pd.DataFrame(events)
        grouped = edf.groupby('side')['volume'].sum()
        grouped.plot(kind='bar', ax=ax6, color=['green','red'])
        ax6.set_title("Total Liquidation Volume by Side")
    else:
        ax6.text(0.5,0.5,"No events yet", ha='center')

    plt.tight_layout()
    plt.show()

# ---- WebSocket handling ----
async def main():
    global trade_counter, last_book_poll
    url = f"wss://stream.binance.us:9443/ws/{SYMBOL}@trade"
    print(f"Connecting to {url} ...")
    async with websockets.connect(url, ping_interval=20) as ws:
        print("Connected. Listening for trades...")
        while True:
            msg = await ws.recv()
            trade = json.loads(msg)
            # trade has fields: e, E, s, t, p, q, b, a, T, m, M
            price = float(trade['p'])
            qty = float(trade['q'])
            side = 'SELL' if trade['m'] else 'BUY'
            ts = datetime.utcnow().isoformat()  # UTC
            trades.append({'ts': ts, 'price': price, 'qty': qty, 'side': side})
            update_price_buckets(price, qty, 'BULLISH' if side=='BUY' else 'BEARISH')

            # detect liquidation-like event
            is_liq, ev = detect_liquidation_like(trade)
            if is_liq:
                print(f"{now_str()} LIQ event: {ev}")

            # Build features for River model
            # Use features: qty, recent_vol_sum(last 10), price_return(last 5), orderbook imbalance (most recent)
            recent_prices = [t['price'] for t in list(trades)[-20:]]
            recent_qty = [t['qty'] for t in list(trades)[-20:]]
            vol_sum_10 = sum(recent_qty[-10:]) if len(recent_qty)>=1 else 0.0
            pr_return = (recent_prices[-1] - recent_prices[0]) / (recent_prices[0] + 1e-9) if len(recent_prices)>1 else 0.0
            # latest imbalance
            last_imb = orderbook_snapshots[-1]['imbalance'] if len(orderbook_snapshots)>0 else 0.0

            X = {
                'qty': qty,
                'vol_sum_10': vol_sum_10,
                'ret_20': pr_return,
                'imbalance': last_imb
            }

            # label is 1 if trade is detected as liquidation-like now
            y = 1 if is_liq else 0

            # prediction before learning
            prob = None
            try:
                # River predict_one safe: returns float prob for LogisticRegression (usually)
                pred = model.predict_proba_one(X)
                if isinstance(pred, dict):
                    # logistic returns {0:prob0,1:prob1}
                    prob = pred.get(1, 0.0)
                else:
                    # fallback
                    prob = float(pred)
            except Exception:
                prob = 0.0

            # store prediction
            prediction_series.append((now_str(), prob))

            # learn online
            model.learn_one(X, y)
            # update metrics
            metric_auc.update(y, prob)
            metric_logloss.update(y, prob)

            trade_counter += 1

            # Poll orderbook occasionally (REST)
            nowt = time.time()
            if nowt - last_book_poll > BOOK_POLL_SECONDS:
                last_book_poll = nowt
                fetch_orderbook_snapshot()

            # periodic plotting
            if trade_counter % PLOT_UPDATE_EVERY == 0:
                clear_output = None
                update_plots()
                print(f"Model AUC: {metric_auc.get():.4f} | LogLoss: {metric_logloss.get():.4f} | Total events:{len(events)}")

# Run the WebSocket listener in Colab (await main)
await main()
