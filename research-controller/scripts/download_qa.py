"""
Download QA datasets (SQuAD and BoolQ) for evaluation and save to data/ directory.

Usage:
    python download_qa.py
"""

import os
import json
import json
import os
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

def save_dataset(ds, name: str):
    """
    Save a HuggingFace Dataset to a JSONL file under the data/ directory.
    """
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, f"{name}.jsonl")
    with open(output_path, "w") as f:
        for record in ds:
            f.write(json.dumps(record) + "\n")
    print(f"Saved {len(ds)} examples to {output_path}")

def main():
    """
    Download SQuAD and BoolQ validation splits via huggingface_hub and save as JSONL.
    """
    # SQuAD: stored as parquet under plain_text/
    squad_parquet = hf_hub_download(
        repo_id="squad", repo_type="dataset",
        filename="plain_text/validation-00000-of-00001.parquet"
    )
    squad_table = pq.read_table(squad_parquet)
    squad_records = squad_table.to_pylist()
    save_dataset(squad_records, "squad")

    # BoolQ: stored as parquet under data/
    boolq_parquet = hf_hub_download(
        repo_id="boolq", repo_type="dataset",
        filename="data/validation-00000-of-00001.parquet"
    )
    boolq_table = pq.read_table(boolq_parquet)
    boolq_records = boolq_table.to_pylist()
    save_dataset(boolq_records, "boolq")

if __name__ == "__main__":
    main()