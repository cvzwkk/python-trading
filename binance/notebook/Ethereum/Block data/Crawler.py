
#!/usr/bin/env python3
import time
import os
from web3 import Web3

# Connect to Ethereum public node (no API key required)
RPC_URL = "https://ethereum.publicnode.com"  # Public Ethereum RPC
w3 = Web3(Web3.HTTPProvider(RPC_URL))

def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_block():
    return w3.eth.get_block('latest', full_transactions=True)

def get_pending_tx_count():
    pending_block = w3.eth.get_block('pending', full_transactions=False)
    return len(pending_block['transactions'])

def calculate_total_eth(block):
    total_wei = 0
    for tx in block['transactions']:
        gas_used = tx['gas']
        gas_price = tx['gasPrice']
        total_wei += gas_used * gas_price
    total_eth = w3.from_wei(total_wei, 'ether')
    total_eth += 2  # Post-merge block reward for validators
    return total_eth

def live_monitor(refresh_sec=5):
    while True:
        try:
            clear_console()

            block = get_latest_block()
            block_number = block['number']
            total_eth = calculate_total_eth(block)
            pending_count = get_pending_tx_count()

            print("=== Ethereum Live Monitor (Web3 - Public Node) ===")
            print(f"Latest Block Number : {block_number}")
            print(f"Total ETH in Block  : {total_eth:.6f} ETH (reward + fees)")
            print(f"Pending TXs (mempool): {pending_count}")

            print(f"\nRefreshing in {refresh_sec} seconds...")
            time.sleep(refresh_sec)

        except KeyboardInterrupt:
            print("\nExiting live monitor...")
            break
        except Exception as e:
            print(f"Error: {e}, retrying in {refresh_sec} seconds...")
            time.sleep(refresh_sec)

if __name__ == "__main__":
    if not w3.is_connected():
        print("Failed to connect to Ethereum node.")
    else:
        live_monitor(refresh_sec=5)
