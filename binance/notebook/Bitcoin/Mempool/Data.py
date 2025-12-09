
#!/usr/bin/env python3
import requests
import time
import os

BASE = "https://mempool.space/api"

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_block_total_btc(block_height):
    """Get total BTC in a mined block (block reward + fees)"""
    block_hash = requests.get(f"https://blockstream.info/api/block-height/{block_height}").text
    txs = requests.get(f"https://blockstream.info/api/block/{block_hash}/txs").json()
    coinbase_tx = txs[0]
    total_sats = sum(vout['value'] for vout in coinbase_tx['vout'])
    total_btc = total_sats / 1e8
    return total_btc

def get_mempool_stats():
    url = f"{BASE}/mempool"
    return requests.get(url).json()

def get_recommended_fees():
    url = f"{BASE}/v1/fees/recommended"
    return requests.get(url).json()

def get_mempool_transactions(limit=50):
    url = f"{BASE}/mempool/recent"
    txs = requests.get(url).json()
    return txs[:limit]

def live_monitor(refresh_sec=5):
    while True:
        try:
            clear_console()

            latest_height = int(requests.get("https://blockstream.info/api/blocks/tip/height").text)
            total_btc = get_block_total_btc(latest_height)
            stats = get_mempool_stats()
            fees = get_recommended_fees()
            txs = get_mempool_transactions(10)

            print("=== Bitcoin Mempool Live Monitor ===")
            print(f"Latest Block Height : {latest_height}")
            print(f"Total BTC in Block  : {total_btc:.8f} BTC")

            print("\n--- MEMPOOL STATS ---")
            print(f"Unconfirmed TXs     : {stats.get('count')}")
            print(f"Mempool Size (vB)   : {stats.get('vsize')}")
            print(f"Mempool Total Fees  : {stats.get('total_fee')} sat")

            print("\n--- RECOMMENDED FEES ---")
            print(f"Fastest (next block): {fees.get('fastestFee')} sat/vB")
            print(f"Half hour           : {fees.get('halfHourFee')} sat/vB")
            print(f"Hour                : {fees.get('hourFee')} sat/vB")
            print(f"Minimum             : {fees.get('minimumFee')} sat/vB")

            print("\n--- SAMPLE UNCONFIRMED TXs (10) ---")
            for tx in txs:
                print(f"- TXID: {tx['txid']}  |  Fee: {tx['fee']} sat  |  vsize: {tx['vsize']}")

            print(f"\nRefreshing in {refresh_sec} seconds...")
            time.sleep(refresh_sec)

        except KeyboardInterrupt:
            print("\nExiting live monitor...")
            break
        except Exception as e:
            print(f"Error: {e}, retrying in {refresh_sec} seconds...")
            time.sleep(refresh_sec)

if __name__ == "__main__":
    live_monitor(refresh_sec=1)
