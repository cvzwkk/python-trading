
#!/usr/bin/env python3
import os
import time
import requests
from web3 import Web3

# ============================================================
# CONFIG
# ============================================================
BSC_RPC = "https://bsc-dataseed.binance.org/"
w3 = Web3(Web3.HTTPProvider(BSC_RPC))

PANCAKE_FACTORY = Web3.to_checksum_address("0xca143ce32fe78f1f7019d7d551a6402fc5350c73")

FACTORY_ABI = [
    {"inputs":[{"name":"tokenA","type":"address"},
               {"name":"tokenB","type":"address"}],
     "name":"getPair",
     "outputs":[{"name":"pair","type":"address"}],
     "stateMutability":"view",
     "type":"function"}
]

PAIR_ABI = [
    {"constant":True,"inputs":[],"name":"getReserves",
     "outputs":[{"name":"reserve0","type":"uint112"},
                {"name":"reserve1","type":"uint112"},
                {"name":"blockTimestampLast","type":"uint32"}],
     "stateMutability":"view","type":"function"}
]

# Tokens
BTCB = Web3.to_checksum_address("0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c")
WBNB = Web3.to_checksum_address("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c")

# ============================================================
# HELPERS
# ============================================================
def clear():
    os.system("cls" if os.name == "nt" else "clear")

factory = w3.eth.contract(address=PANCAKE_FACTORY, abi=FACTORY_ABI)

def get_pair_address():
    try:
        pair = factory.functions.getPair(BTCB, WBNB).call()
        if int(pair, 16) == 0:
            return None
        return pair
    except:
        return None

def get_reserves(pair):
    try:
        contract = w3.eth.contract(address=pair, abi=PAIR_ABI)
        r0, r1, _ = contract.functions.getReserves().call()
        return r0, r1
    except:
        return None, None

def get_binance_price(symbol):
    """Fetch Binance price with retry, returns None if fails"""
    try:
        r = requests.get(f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}", timeout=3)
        return float(r.json()['price'])
    except:
        return None

def simulate_swap_input(reserve_in, reserve_out, amount_in):
    """PancakeSwap formula for output amount"""
    try:
        amount_in_with_fee = amount_in * 997
        return (amount_in_with_fee * reserve_out) / (reserve_in * 1000 + amount_in_with_fee)
    except:
        return None

def usd(n):
    try:
        return f"${n:,.2f}"
    except:
        return str(n)

def num(n):
    try:
        return f"{n:,.6f}"
    except:
        return str(n)

# ============================================================
# MAIN LIVE ANALYZER
# ============================================================
def live_monitor(interval=5):
    pair = get_pair_address()
    if pair is None:
        print("⚠️ BTCB/WBNB pair not found on PancakeSwap!")
        return

    print(f"Found PancakeSwap BTCB/WBNB pair: {pair}")
    time.sleep(2)

    while True:
        try:
            clear()
            print("=== BTCB / WBNB — PancakeSwap Deep Analyzer ===\n")

            r0, r1 = get_reserves(pair)
            if r0 is None or r1 is None:
                print("⚠️ Could not fetch pair reserves. Retrying...")
                time.sleep(interval)
                continue

            # Token order
            reserve_btcb, reserve_wbnb = r0 / 1e18, r1 / 1e18

            price_btcb = reserve_wbnb / reserve_btcb if reserve_btcb else None
            price_wbnb = reserve_btcb / reserve_wbnb if reserve_wbnb else None

            # External prices with fallback
            btc_usdt = get_binance_price("BTCUSDT") or 0
            bnb_usdt = get_binance_price("BNBUSDT") or 0
            onchain_btcb_usd = price_btcb * bnb_usdt if price_btcb and bnb_usdt else 0

            # Output
            print(f"Pair Address: {pair}")
            print(f"\n--- Reserves ---")
            print(f"BTCB reserve : {num(reserve_btcb)}")
            print(f"WBNB reserve : {num(reserve_wbnb)}")

            print("\n--- Prices ---")
            print(f"1 BTCB = {num(price_btcb)} WBNB")
            print(f"1 WBNB = {num(price_wbnb)} BTCB")

            print("\n--- USD Prices ---")
            print(f"BTC Price Binance:  {usd(btc_usdt)}")
            print(f"BNB Price Binance:  {usd(bnb_usdt)}")
            print(f"BTCB On-chain USD:  {usd(onchain_btcb_usd)}")

            diff = ((onchain_btcb_usd - btc_usdt) / btc_usdt * 100) if btc_usdt else 0
            if abs(diff) > 0.5:
                print(f"\n⚠️ Arbitrage difference detected: {diff:.2f}%")

            # Slippage simulation
            print("\n--- Slippage Simulation ---")
            if not btc_usdt or not bnb_usdt:
                print("⚠️ Could not fetch external prices, skipping slippage simulation.")
            else:
                for usd_amount in [100, 1000, 10000]:
                    bnb_amount = usd_amount / bnb_usdt
                    out = simulate_swap_input(reserve_wbnb * 1e18, reserve_btcb * 1e18, bnb_amount * 1e18)
                    if out:
                        out /= 1e18
                        ideal = bnb_amount / price_btcb if price_btcb else 0
                        slip = ((ideal - out) / ideal * 100) if ideal else 0
                        print(f"Buying BTCB with {usd(usd_amount)}:")
                        print(f" • BTCB Received: {num(out)}")
                        print(f" • Slippage: {slip:.4f}%")
                    else:
                        print(f" • Could not simulate swap for {usd(usd_amount)} USD")

            print(f"\nRefreshing in {interval} seconds...")
            time.sleep(interval)

        except KeyboardInterrupt:
            print("\nExiting monitor...")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}, retrying in {interval} seconds...")
            time.sleep(interval)

# ============================================================
# RUN
# ============================================================
if __name__ == "__main__":
    if not w3.is_connected():
        print("Error: could not connect to BSC RPC.")
    else:
        live_monitor(interval=5)
