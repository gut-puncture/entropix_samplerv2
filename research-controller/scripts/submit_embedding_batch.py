#!/usr/bin/env python3
"""
Submits a specified JSONL file to the OpenAI Batch API for embedding.

Usage:
    python scripts/submit_embedding_batch.py <input_jsonl_file>

Example:
    python scripts/submit_embedding_batch.py data/embed_tasks_part_1.jsonl

Requires OPENAI_API_KEY to be set in a .env file or environment variables.
Logs submitted job IDs to data/batch_jobs.log
"""

import os
import sys
import time
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
import argparse # Use argparse for better argument handling

def main():
    parser = argparse.ArgumentParser(description="Submit an embedding batch job to OpenAI.")
    parser.add_argument("input_file", help="Path to the input JSONL file for the batch job.")
    args = parser.parse_args()

    batch_input_file_path = Path(args.input_file)
    log_file_path = Path("data") / "batch_jobs.log"

    # Load environment variables from .env file
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        print("Please create a .env file with OPENAI_API_KEY=<your_key> or set the environment variable.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if not batch_input_file_path.exists():
        print(f"Error: Batch input file not found at {batch_input_file_path}")
        sys.exit(1)

    model_name = "text-embedding-3-large"
    print(f"Preparing to submit batch job for model: {model_name}")
    print(f"Input file: {batch_input_file_path}")

    try:
        # 1. Upload the batch file
        print(f"Uploading {batch_input_file_path.name}...")
        with open(batch_input_file_path, "rb") as f:
            batch_input_file = client.files.create(
                file=f,
                purpose="batch"
            )
        print(f"File uploaded successfully. File ID: {batch_input_file.id}")

        # Wait briefly for the file to be ready for processing
        print("Waiting for file processing...")
        time.sleep(10) # Increased wait time slightly

        # 2. Create the batch job
        print("Creating batch job...")
        batch_job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/embeddings",
            completion_window="24h",  # Maximum time allowed for the job
            metadata={
                "description": f"Embedding job part for {batch_input_file_path.name}"
            }
        )
        print("Batch job created successfully!")
        print(f"Batch Job ID: {batch_job.id}")
        print(f"Status: {batch_job.status}")

        # Log the job ID and input file
        try:
            with log_file_path.open("a") as log_f:
                log_f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{batch_job.id},{batch_input_file.id},{batch_input_file_path.name}\n")
            print(f"Logged job ID to {log_file_path}")
        except Exception as log_e:
            print(f"Warning: Failed to log job ID to {log_file_path}: {log_e}")

        print("\n--- Monitoring Instructions ---")
        print("You can monitor the job status using the OpenAI CLI or API, or the check script.")
        print(f"Check Script: python scripts/check_embedding_batch.py {batch_job.id}")
        print("Once the status is 'completed', retrieve the output file content.")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Consider adding logic here to cancel the uploaded file if the batch creation failed
        # if 'batch_input_file' in locals() and batch_input_file:
        #    try:
        #        client.files.delete(batch_input_file.id)
        #        print(f"Cleaned up uploaded file {batch_input_file.id}")
        #    except Exception as del_e:
        #        print(f"Warning: Failed to clean up file {batch_input_file.id}: {del_e}")

if __name__ == "__main__":
    main() 