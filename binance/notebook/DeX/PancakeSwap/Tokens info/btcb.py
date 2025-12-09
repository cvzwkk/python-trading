
#!/usr/bin/env python3
import os
import time
from web3 import Web3

# ------------------------
# CONFIGURATION
# ------------------------

BSC_RPC = "https://bsc-dataseed.binance.org/"
w3 = Web3(Web3.HTTPProvider(BSC_RPC))

# PancakeSwap V2 Router
PANCAKESWAP_ROUTER = Web3.to_checksum_address("0x10ED43C718714eb63d5aA57B78B54704E256024E")
PANCAKESWAP_FACTORY = Web3.to_checksum_address("0xca143ce32fe78f1f7019d7d551a6402fc5350c73")

# ------------------------
# TOKENS
# ------------------------
TOKENS = {
    "BTCB": Web3.to_checksum_address("0x7130d2A12B9BCbFAe4f2634d864A1Ee1Ce3Ead9c"),
    "WBNB": Web3.to_checksum_address("0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c"),
}

# Pairs to monitor
PAIRS = [
    ("BTCB", "WBNB"),
]

# Amount to simulate swap (in token0 units)
SIMULATE_AMOUNTS = {
    "BTCB": 0.01,
    "WBNB": 0.1,
    "ETH": 0.1
}

# ------------------------
# ABIs
# ------------------------
FACTORY_ABI = [
    {"inputs":[{"internalType":"address","name":"tokenA","type":"address"},
               {"internalType":"address","name":"tokenB","type":"address"}],
     "name":"getPair",
     "outputs":[{"internalType":"address","name":"pair","type":"address"}],
     "stateMutability":"view",
     "type":"function"}
]

PAIR_ABI = [
    {"constant":True,"inputs":[],"name":"getReserves",
     "outputs":[{"internalType":"uint112","name":"_reserve0","type":"uint112"},
                {"internalType":"uint112","name":"_reserve1","type":"uint112"},
                {"internalType":"uint32","name":"_blockTimestampLast","type":"uint32"}],
     "payable":False,"stateMutability":"view","type":"function"}
]

ROUTER_ABI = [
    {"inputs":[
        {"internalType":"uint256","name":"amountIn","type":"uint256"},
        {"internalType":"address[]","name":"path","type":"address[]"}
    ],
     "name":"getAmountsOut",
     "outputs":[{"internalType":"uint256[]","name":"","type":"uint256[]"}],
     "stateMutability":"view",
     "type":"function"}
]

# ------------------------
# FUNCTIONS
# ------------------------
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_pair_address(tokenA, tokenB):
    factory = w3.eth.contract(address=PANCAKESWAP_FACTORY, abi=FACTORY_ABI)
    return factory.functions.getPair(TOKENS[tokenA], TOKENS[tokenB]).call()

def get_reserves(pair_address):
    pair = w3.eth.contract(address=pair_address, abi=PAIR_ABI)
    return pair.functions.getReserves().call()

def get_price(tokenA, tokenB, pair_address):
    reserve0, reserve1, _ = get_reserves(pair_address)
    price = (reserve1 / 1e18) / (reserve0 / 1e18)
    return price

def simulate_swap(tokenA, tokenB, amount_in):
    router = w3.eth.contract(address=PANCAKESWAP_ROUTER, abi=ROUTER_ABI)
    amount_in_wei = int(amount_in * 1e18)
    path = [TOKENS[tokenA], TOKENS[tokenB]]
    try:
        amounts_out = router.functions.getAmountsOut(amount_in_wei, path).call()
        amount_out = amounts_out[-1] / 1e18
        # Slippage risk: difference from price*amount
        pair_address = get_pair_address(tokenA, tokenB)
        price = get_price(tokenA, tokenB, pair_address)
        expected_out = amount_in * price
        slippage_percent = ((expected_out - amount_out) / expected_out) * 100
        return amount_out, slippage_percent
    except Exception:
        return None, None

def live_monitor(refresh_sec=5):
    # Pre-fetch pair addresses
    pair_addresses = {}
    for tokenA, tokenB in PAIRS:
        pair_addr = get_pair_address(tokenA, tokenB)
        if pair_addr == "0x0000000000000000000000000000000000000000":
            print(f"Pair {tokenA}/{tokenB} does not exist. Skipping.")
        else:
            pair_addresses[(tokenA, tokenB)] = pair_addr

    if not pair_addresses:
        print("No valid pairs to monitor. Exiting.")
        return

    while True:
        try:
            clear_console()
            print("=== PancakeSwap Live Monitor with Slippage ===")

            for (tokenA, tokenB), pair_addr in pair_addresses.items():
                price = get_price(tokenA, tokenB, pair_addr)
                swap_amount = SIMULATE_AMOUNTS.get(tokenA, 0.01)
                amount_out, slippage = simulate_swap(tokenA, tokenB, swap_amount)
                print(f"\nPair: {tokenA}/{tokenB}")
                print(f"Pair Address: {pair_addr}")
                print(f"Current Price ({tokenB} per {tokenA}): {price:.6f}")
                if amount_out:
                    print(f"Simulated swap: {swap_amount} {tokenA} -> {amount_out:.6f} {tokenB}")
                    print(f"Estimated slippage: {slippage:.4f}%")
                else:
                    print("Could not simulate swap (maybe low liquidity)")

            print(f"\nRefreshing in {refresh_sec} seconds...")
            time.sleep(refresh_sec)

        except KeyboardInterrupt:
            print("\nExiting monitor...")
            break
        except Exception as e:
            print(f"Error: {e}, retrying in {refresh_sec} seconds...")
            time.sleep(refresh_sec)

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    if not w3.is_connected():
        print("Failed to connect to BSC node.")
    else:
        live_monitor(refresh_sec=5)
