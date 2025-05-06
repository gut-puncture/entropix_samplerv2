#!/usr/bin/env python3
"""
Split prompts from data/prompts.json into sentences and save to data/prompts_sentences.jsonl
"""
import json
from pathlib import Path

def split_text(text):
    # Naive split on '.', '!', '?'
    for delim in ['.', '!', '?']:
        text = text.replace(delim, '.')
    return [s.strip() for s in text.split('.') if s.strip()]

def main():
    data_path = Path("data")
    prompts_file = data_path / "prompts.json"
    output_file = data_path / "prompts_sentences.jsonl"

    prompts = json.loads(prompts_file.read_text())
    count = 0
    with output_file.open("w") as f:
        for p in prompts:
            for sent in split_text(p):
                f.write(json.dumps({"sentence": sent}) + "\n")
                count += 1
    print(f"Saved {count} sentences to {output_file}")

if __name__ == "__main__":
    main() 