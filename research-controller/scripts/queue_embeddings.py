#!/usr/bin/env python3
"""
Queue sentences from prompts_sentences.jsonl for OpenAI embedding batch requests,
creating a file in the correct JSONL format.
"""
import json
from pathlib import Path

def main():
    data_path = Path("data")
    in_file = data_path / "prompts_sentences.jsonl"
    out_file = data_path / "embed_tasks_formatted.jsonl" # New name to avoid confusion
    model_name = "text-embedding-3-large"
    endpoint = "/v1/embeddings"
    count = 0
    
    print(f"Reading sentences from {in_file}...")
    if not in_file.exists():
        print(f"Error: Input sentence file not found at {in_file}")
        print("Please run scripts/split_prompts.py first.")
        return

    print(f"Writing formatted batch requests to {out_file}...")
    with in_file.open() as fin, out_file.open("w") as fout:
        for i, line in enumerate(fin):
            try:
                sentence = json.loads(line)["sentence"]
                # Create the JSON object required by the Batch API for embeddings
                batch_request = {
                    "custom_id": f"request-{i+1}",
                    "method": "POST",
                    "url": endpoint,
                    "body": {
                        "input": sentence,
                        "model": model_name
                    }
                }
                fout.write(json.dumps(batch_request) + "\n")
                count += 1
            except json.JSONDecodeError:
                print(f"Warning: Skipping invalid JSON line {i+1} in {in_file}")
            except KeyError:
                print(f"Warning: Skipping line {i+1} in {in_file} - missing 'sentence' key")
            
    print(f"Created {count} formatted batch requests in {out_file}")

if __name__ == "__main__":
    main() 