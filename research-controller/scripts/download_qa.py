"""
Download QA datasets (SQuAD and BoolQ) for evaluation and save to data/ directory.

Usage:
    python download_qa.py
"""

import os
import json
from datasets import load_dataset

def save_dataset(ds, name):
    # Ensure data directory exists
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", f"{name}.jsonl")
    with open(output_path, "w") as f:
        for example in ds:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(ds)} examples to {output_path}")

def main():
    # Load validation splits
    squad = load_dataset("squad", split="validation")
    boolq = load_dataset("boolq", split="validation")
    save_dataset(squad, "squad")
    save_dataset(boolq, "boolq")

if __name__ == "__main__":
    main()