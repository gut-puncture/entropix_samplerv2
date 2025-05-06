#!/usr/bin/env python3
"""
Splits a large JSONL file into smaller parts for OpenAI Batch API.

Usage:
    python scripts/split_batch_input.py <input_file> <max_lines_per_file>

Example:
    python scripts/split_batch_input.py data/embed_tasks.jsonl 50000
"""

import sys
import os
from pathlib import Path

def main():
    if len(sys.argv) != 3:
        print("Usage: python scripts/split_batch_input.py <input_file> <max_lines_per_file>")
        sys.exit(1)

    input_file_path = Path(sys.argv[1])
    try:
        max_lines = int(sys.argv[2])
    except ValueError:
        print("Error: max_lines_per_file must be an integer.")
        sys.exit(1)

    if not input_file_path.exists():
        print(f"Error: Input file not found at {input_file_path}")
        sys.exit(1)

    if max_lines <= 0:
        print("Error: max_lines_per_file must be positive.")
        sys.exit(1)

    file_index = 1
    line_count_in_current_file = 0
    output_file = None
    output_f = None

    base_name = input_file_path.stem
    extension = input_file_path.suffix
    output_dir = input_file_path.parent

    print(f"Splitting {input_file_path} into chunks of max {max_lines} lines...")

    try:
        with input_file_path.open('r', encoding='utf-8') as infile:
            for line in infile:
                # If current file is full or it's the first line, start a new file
                if line_count_in_current_file == max_lines or output_f is None:
                    if output_f:
                        output_f.close()
                        print(f"  Wrote {line_count_in_current_file} lines to {output_file}")
                    
                    output_file = output_dir / f"{base_name}_part_{file_index}{extension}"
                    output_f = output_file.open('w', encoding='utf-8')
                    print(f"Creating {output_file}...")
                    file_index += 1
                    line_count_in_current_file = 0 # Reset line count for the new file

                output_f.write(line)
                line_count_in_current_file += 1

        if output_f:
            output_f.close()
            print(f"  Wrote {line_count_in_current_file} lines to {output_file}") # Print count for the last file
        
        print("Splitting complete.")

    except Exception as e:
        print(f"An error occurred during splitting: {e}")
        if output_f and not output_f.closed:
            output_f.close() # Ensure file is closed on error

if __name__ == "__main__":
    main() 