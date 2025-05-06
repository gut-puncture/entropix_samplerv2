#!/usr/bin/env python3
"""
Checks the status of an OpenAI Batch API job.

Usage:
    python scripts/check_embedding_batch.py <batch_job_id>

Requires OPENAI_API_KEY to be set in a .env file or environment variables.
"""

import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_embedding_batch.py <batch_job_id>")
        sys.exit(1)

    batch_job_id = sys.argv[1]

    # Load environment variables from .env file
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    print(f"Checking status for Batch Job ID: {batch_job_id}")

    try:
        batch_job = client.batches.retrieve(batch_job_id)

        print(f"Status: {batch_job.status}")

        if batch_job.status == 'completed':
            print(f"Input File ID: {batch_job.input_file_id}")
            print(f"Output File ID: {batch_job.output_file_id}")
            print(f"Error File ID: {batch_job.error_file_id}") # Usually None if successful
            print("\nJob completed! You can now download the output file.")
            print("Example download command (using OpenAI Python library, create a script):")
            print(f"  content = client.files.content(\"{batch_job.output_file_id}\").read()")
            print(f"  with open(\"data/embeddings_output.jsonl\", \"wb\") as f:")
            print(f"      f.write(content)")

        elif batch_job.status in ['validating', 'in_progress', 'queued']:
            print("Job is still processing.")
        elif batch_job.status in ['failed', 'cancelled', 'expired']:
            print("Job did not complete successfully.")
            print(f"Error File ID: {batch_job.error_file_id}")
            print("Check the error file for details.")
        else:
            print("Unknown status.")

    except Exception as e:
        print(f"\nAn error occurred while checking the batch job: {e}")

if __name__ == "__main__":
    main() 