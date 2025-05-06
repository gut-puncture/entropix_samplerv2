#!/usr/bin/env python3
"""
Load raw WikiText-2 training data, filter articles > 20 words,
and save them to data/prompts.json
"""
import json
import re
from pathlib import Path


def main():
    data_path = Path("data")
    raw_file = data_path / "wiki.train.raw"
    output_file = data_path / "prompts.json"

    if not raw_file.exists():
        print(f"Error: Raw data file not found at {raw_file}")
        print("Please download it first, e.g., using:")
        print("curl -L -o data/wiki.train.raw https://cosmo.zip/pub/datasets/wikitext-2-raw/wiki.train.raw")
        return

    print(f"Reading raw data from {raw_file}...")
    raw_text = raw_file.read_text(encoding='utf-8')

    # Use a more flexible regex for separators (e.g., " \n = ... = \n ")
    # Split and keep delimiters
    parts = re.split(r'(\n\s*=\s*[^=]+\s*=\s*\n)', raw_text)

    # The first part might be empty or content before the first header
    # Subsequent parts alternate between separator and content
    articles = []
    # Start from index 1 if the split starts with a separator, or 0 if it starts with text
    start_index = 1 if parts and re.match(r'\n\s*=\s*[^=]+\s*=\s*\n', parts[0]) is None else 0

    if not parts or len(parts) < (start_index + 2):
         # If no separators found, treat the whole file as one article (or handle as error)
         print("Warning: No article separators found. Treating entire file as one potential prompt.")
         if parts and parts[0].strip(): # Check if there's any text at all
            articles.append(parts[0].strip()) 
    else:
        # Iterate through pairs of (separator, content)
        for i in range(start_index + 1, len(parts), 2):
            # Combine the separator (title) with its content
            article_text = parts[i-1] + parts[i]
            articles.append(article_text.strip())

    print(f"Found {len(articles)} potential articles.")

    prompts = []
    for article in articles:
        # Clean up extra whitespace and newlines within the article
        cleaned_article = ' '.join(article.split())
        if len(cleaned_article.split()) > 20:
            prompts.append(cleaned_article)

    with output_file.open("w") as f:
        json.dump(prompts, f, indent=2)

    print(f"Saved {len(prompts)} prompts (> 20 words) to {output_file}")

if __name__ == "__main__":
    main()