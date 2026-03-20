"""
Mercari Listing Normalization Pipeline
Uses OpenAI Batch API for cost-efficient bulk normalization (~50% discount).
Supports two prompt strategies and checkpointing every 1000 rows.

Usage:
    python normalize_mercari.py --input train.tsv --prompt 1 --output train_normalized.tsv
    python normalize_mercari.py --input train.tsv --prompt 2 --output train_normalized.tsv
"""

import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
CHECKPOINT_INTERVAL = 1000   # Save progress every N rows
BATCH_CHUNK_SIZE = 10_000    # Submit in chunks to make failures recoverable
POLL_INTERVAL = 30           # Seconds between batch status checks

PROMPTS = {
    1: (
        "You are a text normalizer for a consumer marketplace price prediction system. "
        "Your task is to rewrite product listings into clear, structured, and complete descriptions. "
        "Expand all abbreviations and acronyms. Correct spelling errors. Normalize condition descriptions "
        "(e.g. 'gr8 cond' → 'great condition'). Improve sentence structure and clarity. "
        "Incorporate any metadata that is implied but not stated explicitly. "
        "IMPORTANT: Preserve all factual details exactly — brand names, model numbers, storage sizes, "
        "colors, and specifications must not be altered, invented, or removed. "
        "Return only the rewritten listing text, nothing else."
    ),
    2: (
        "You are a text normalizer for price prediction. Rewrite the product listing to be clear and "
        "well-structured: expand abbreviations, fix spelling, and normalize informal language. "
        "Do not change or invent any factual details. Return only the rewritten text."
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_text(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("category_name")) and str(row["category_name"]).strip():
        parts.append(f"Category: {row['category_name']}.")
    if pd.notna(row.get("name")) and str(row["name"]).strip():
        parts.append(str(row["name"]))
    if pd.notna(row.get("item_description")) and str(row["item_description"]).strip():
        parts.append(str(row["item_description"]))
    return " ".join(parts).strip()

def load_checkpoint(checkpoint_path: Path) -> dict:
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint_path: Path, results: dict):
    with open(checkpoint_path, "w") as f:
        json.dump(results, f)
    print(f"  Checkpoint saved — {len(results)} rows complete.")


def build_batch_requests(rows: list[dict], prompt_text: str) -> list[dict]:
    """Format rows as JSONL batch requests."""
    requests = []
    for row in rows:
        requests.append({
            "custom_id": str(row["train_id"]),
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "max_tokens": 300,
                "temperature": 0.0,   # Deterministic — important for reproducibility
                "messages": [
                    {"role": "system", "content": prompt_text},
                    {"role": "user",   "content": row["text"]},
                ],
            },
        })
    return requests


def submit_batch(client: OpenAI, requests: list[dict], tmp_dir: Path) -> str:
    """Write JSONL file, upload it, and submit a batch job. Returns batch_id."""
    jsonl_path = tmp_dir / "batch_input.jsonl"
    with open(jsonl_path, "w") as f:
        for req in requests:
            f.write(json.dumps(req) + "\n")

    with open(jsonl_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Batch submitted: {batch.id}")
    return batch.id


def poll_batch(client: OpenAI, batch_id: str) -> str:
    """Poll until batch completes. Returns output_file_id."""
    while True:
        batch = client.batches.retrieve(batch_id)
        status = batch.status
        print(f"  Batch {batch_id} status: {status} "
              f"({batch.request_counts.completed}/{batch.request_counts.total})")
        if status == "completed":
            return batch.output_file_id
        elif status in ("failed", "expired", "cancelled"):
            raise RuntimeError(f"Batch {batch_id} ended with status: {status}")
        time.sleep(POLL_INTERVAL)


def download_batch_results(client: OpenAI, output_file_id: str) -> dict[str, str]:
    """Download batch output and return {custom_id: normalized_text}."""
    content = client.files.content(output_file_id).text
    results = {}
    for line in content.strip().split("\n"):
        item = json.loads(line)
        custom_id = item["custom_id"]
        try:
            text = item["response"]["body"]["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError):
            text = None   # Mark failed rows — will be filled with original text
        results[custom_id] = text
    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def normalize(input_path: str, output_path: str, prompt_id: int):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    prompt_text = PROMPTS[prompt_id]

    tmp_dir = Path("tmp_normalization")
    tmp_dir.mkdir(exist_ok=True)
    checkpoint_path = tmp_dir / f"checkpoint_prompt{prompt_id}.json"

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path, sep="\t")

    # Build combined text column the same way your model training does
    df["combined_text"] = df.apply(build_text, axis=1)

    # Load existing checkpoint so we can resume if interrupted
    completed = load_checkpoint(checkpoint_path)
    print(f"Resuming from checkpoint: {len(completed)} rows already done.")

    # Only process rows not yet in checkpoint
    remaining = df[~df["train_id"].astype(str).isin(completed.keys())]
    print(f"Rows remaining: {len(remaining)}")

    # Process in chunks of BATCH_CHUNK_SIZE
    chunks = [
        remaining.iloc[i:i + BATCH_CHUNK_SIZE]
        for i in range(0, len(remaining), BATCH_CHUNK_SIZE)
    ]

    for chunk_idx, chunk in enumerate(chunks):
        print(f"\nChunk {chunk_idx + 1}/{len(chunks)} — {len(chunk)} rows")

        rows = [
            {"train_id": row["train_id"], "text": row["combined_text"]}
            for _, row in chunk.iterrows()
        ]

        requests = build_batch_requests(rows, prompt_text)
        batch_id = submit_batch(client, requests, tmp_dir)
        output_file_id = poll_batch(client, batch_id)
        batch_results = download_batch_results(client, output_file_id)

        # Merge into checkpoint, fall back to original text on failure
        for row in rows:
            sid = str(row["train_id"])
            normalized = batch_results.get(sid)
            if not normalized:
                print(f"  Warning: no output for train_id {sid}, using original.")
                normalized = row["text"]
            completed[sid] = normalized

        save_checkpoint(checkpoint_path, completed)

    # Write final output TSV
    print(f"\nWriting output to {output_path}...")
    df["normalized_text"] = df["train_id"].astype(str).map(completed)

    # Sanity check
    n_missing = df["normalized_text"].isna().sum()
    if n_missing > 0:
        print(f"Warning: {n_missing} rows missing normalized text — filling with original.")
        df["normalized_text"] = df["normalized_text"].fillna(df["combined_text"])

    df.to_csv(output_path, sep="\t", index=False)
    print(f"Done. {len(df)} rows written to {output_path}")
    print(f"Checkpoint retained at {checkpoint_path} for reference.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalize Mercari listings via OpenAI Batch API")
    parser.add_argument("--input",  required=True, help="Input TSV file (e.g. train.tsv)")
    parser.add_argument("--output", required=True, help="Output TSV file")
    parser.add_argument("--prompt", type=int, choices=[1, 2], required=True,
                        help="Prompt strategy: 1 = detailed, 2 = concise")
    args = parser.parse_args()

    normalize(args.input, args.output, args.prompt)