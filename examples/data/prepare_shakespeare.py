#!/usr/bin/env python3
"""
Prepare TinyShakespeare dataset for MetaGen training.

Downloads the dataset, tokenizes with tiktoken GPT-2 BPE,
and saves as binary files (train.bin, val.bin).

Usage:
    python prepare_data.py

This will create:
    - input.txt (raw text)
    - train.bin (tokenized training data, 90%)
    - val.bin (tokenized validation data, 10%)
"""

import os

import numpy as np
import requests
import tiktoken

# Download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), "input.txt")
if not os.path.exists(input_file_path):
    print("Downloading TinyShakespeare dataset...")
    data_url = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    )
    try:
        response = requests.get(data_url, timeout=10)
        response.raise_for_status()
    except requests.Timeout:
        print("Error: Download timed out after 10 seconds.")
        raise SystemExit(1) from None
    except requests.RequestException as exc:
        print(f"Error: Failed to download dataset: {exc}")
        raise SystemExit(1) from None
    with open(input_file_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    print(f"Saved to {input_file_path}")
else:
    print(f"Dataset already exists at {input_file_path}")

with open(input_file_path, encoding="utf-8") as f:
    data = f.read()

n = len(data)
train_data = data[: int(n * 0.9)]
val_data = data[int(n * 0.9) :]

# Encode with tiktoken gpt2 bpe
print("Tokenizing with tiktoken GPT-2 BPE...")
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"Train has {len(train_ids):,} tokens")
print(f"Val has {len(val_ids):,} tokens")

# Export to bin files
script_dir = os.path.dirname(__file__)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_path = os.path.join(script_dir, "train.bin")
val_path = os.path.join(script_dir, "val.bin")

train_ids.tofile(train_path)
val_ids.tofile(val_path)

print(f"Saved train.bin ({len(train_ids):,} tokens) to {train_path}")
print(f"Saved val.bin ({len(val_ids):,} tokens) to {val_path}")
