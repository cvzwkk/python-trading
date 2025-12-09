#!/usr/bin/env python3
import requests
import time

BASE = "https://mempool.space/api"

def get_mempool_stats():
    """Returns mempool size, transaction count, usage."""
    url = f"{BASE}/mempool"
    return requests.get(url).json()

def get_recommended_fees():
    """Returns fast/normal/low fee rates in sat/vB."""
    url = f"{BASE}/v1/fees/recommended"
    return requests.get(url).json()

def get_mempool_transactions(limit=50):
    """Fetch N most recent unconfirmed transactions."""
    url = f"{BASE}/mempool/recent"
    txs = requests.get(url).json()
    return txs[:limit]

def main():
    print("=== Bitcoin Mempool Monitor ===")

    stats = get_mempool_stats()
    fees = get_recommended_fees()

    print("\n--- MEMPOOL STATS ---")
    print(f"Unconfirmed TXs     : {stats.get('count')}")
    print(f"Mempool Size (vB)   : {stats.get('vsize')}")
    print(f"Mempool Total Fees  : {stats.get('total_fee')} sat")

    print("\n--- RECOMMENDED FEES ---")
    print(f"Fastest (next block): {fees.get('fastestFee')} sat/vB")
    print(f"Half hour           : {fees.get('halfHourFee')} sat/vB")
    print(f"Hour                : {fees.get('hourFee')} sat/vB")
    print(f"Minimum             : {fees.get('minimumFee')} sat/vB")

    print("\n--- SAMPLE UNCONFIRMED TXs (showing 10) ---")
    txs = get_mempool_transactions(10)
    for tx in txs:
        print(f"- TXID: {tx['txid']}  |  Fee: {tx['fee']} sat  |  vsize: {tx['vsize']}")

if __name__ == "__main__":
    main()
