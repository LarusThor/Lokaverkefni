"""
Batch Normalization Script — Sequential Mode
Processes one chunk at a time: submit → poll → download → next chunk.
This respects OpenAI's 2,000,000 enqueued token limit by never having
more than one batch job in the queue at once.

Output files (same schema as input, item_description replaced with normalized text):
    train_p1.csv, validation_p1.csv, test_p1.csv
    train_p2.csv, validation_p2.csv, test_p2.csv

Usage:
    python normalize_batch.py

    If interrupted, re-run the same command — already completed chunks are
    skipped automatically via cached result files in batch_work/.

Environment:
    Set your OpenAI API key directly in the client instantiation below.
"""

import json
import math
import time
import pandas as pd
from pathlib import Path
from openai import OpenAI

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gpt-4o-mini"
MAX_TOKENS = 300
BATCH_CHUNK_SIZE = 4400 
POLL_INTERVAL = 60         # seconds between status checks

SPLITS = {
    "train":      "train_sample.csv",
    "validation": "validation_sample.csv",
    "test":       "test_sample.csv",
}

PROMPTS = {
    1: (
        "You are a text normalizer for a consumer marketplace price prediction system. "
        "Your task is to rewrite product listings into clear, fluent prose descriptions. "
        "Expand all abbreviations and acronyms. Correct spelling errors. Normalize condition descriptions "
        "(e.g. 'gr8 cond' → 'great condition'). Improve sentence structure and clarity. "
        "Only interpret vague or non-standard terms when the meaning cannot be understood as written "
        "(e.g. 'skin energy drink' → 'facial moisturizer'). Do not add descriptive language, "
        "use cases, or editorial commentary that is not present in the original listing. "
        "Remove self-promotional seller noise such as follower requests, discount offers for bundles, "
        "and unsolicited personal appeals — keep only information relevant to the product itself. "
        "IMPORTANT: Preserve all factual details exactly — brand names, model numbers, storage sizes, "
        "colors, and specifications must not be altered, invented, or removed. "
        "Preserve the original sentiment and intensity of descriptive language — do not upgrade or "
        "downgrade adjectives (e.g. 'great' must not become 'fantastic', 'okay' must not become 'good'). "
        "Return only the rewritten listing as plain prose, no bullet points or headers."
    ),
    2: (
        "You are a product attribute extractor for a price prediction system. "
        "Given a raw marketplace listing, extract and reformat the information into a structured "
        "template using only details explicitly present or clearly implied by the listing. "
        "Use this exact format:\n"
        "Item: [product type]. Brand: [brand or 'Unknown']. Model: [model/version or 'N/A']. "
        "Color: [color or 'N/A']. Size: [size/storage/dimensions or 'N/A']. "
        "Condition: [condition in standard terms, e.g. 'new', 'like new', 'good', 'fair', 'poor']. "
        "Features: [key specs or attributes, comma-separated, or 'N/A']. "
        "Description: [one sentence summary of the item in plain English].\n"
        "Do not invent, infer beyond what is stated, or include seller notes, shipping info, "
        "or payment warnings. If a field cannot be determined, use 'N/A' or 'Unknown'."
    ),
}

WORK_DIR = Path("batch_work")
OUTPUT_DIR = Path(".")

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


def load_all_splits() -> dict[str, pd.DataFrame]:
    dfs = {}
    for split, filename in SPLITS.items():
        path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Could not find {filename} — make sure it's in the working directory.")
        dfs[split] = pd.read_csv(path)
        print(f"  Loaded {split}: {len(dfs[split])} rows")
    return dfs


def build_all_records(dfs: dict, prompt_num: int) -> list[dict]:
    prompt_text = PROMPTS[prompt_num]
    records = []
    for split, df in dfs.items():
        for _, row in df.iterrows():
            custom_id = f"{split}_{row['train_id']}"
            text = build_text(row)
            records.append({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": MODEL,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "messages": [
                        {"role": "system", "content": prompt_text},
                        {"role": "user",   "content": text},
                    ],
                },
            })
    return records


def write_chunk_jsonl(records: list[dict], prompt_num: int, chunk_idx: int) -> Path:
    WORK_DIR.mkdir(exist_ok=True)
    path = WORK_DIR / f"batch_input_p{prompt_num}_chunk{chunk_idx}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def submit_chunk(client: OpenAI, jsonl_path: Path, prompt_num: int, chunk_idx: int) -> str:
    print(f"    Uploading {jsonl_path.name}...")
    with open(jsonl_path, "rb") as f:
        upload = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"prompt": str(prompt_num), "chunk": str(chunk_idx)},
    )
    print(f"    Batch submitted: {batch.id}  (status: {batch.status})")

    # Cache the batch ID in case of interruption
    (WORK_DIR / f"batch_id_p{prompt_num}_chunk{chunk_idx}.txt").write_text(batch.id)
    return batch.id


def poll_until_done(client: OpenAI, batch_id: str, label: str) -> str | None:
    """
    Block until batch completes. Returns output_file_id, or None if the batch
    stalled (no progress for STALL_TIMEOUT seconds) and was cancelled.
    Partial results are still available via the output file on a cancelled batch.
    """
    STALL_TIMEOUT = 30 * 60  # 30 minutes with no progress → cancel

    print(f"    Polling {label}...")
    last_completed = -1
    stall_since = None

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(f"      status={batch.status}  completed={counts.completed}  failed={counts.failed}  total={counts.total}")

        if batch.status == "completed":
            return batch.output_file_id

        if batch.status in ("failed", "expired", "cancelled"):
            # Return partial output file if available, else None
            if batch.output_file_id:
                print(f"    Batch ended with status '{batch.status}' — partial results available.")
                return batch.output_file_id
            raise RuntimeError(f"Batch {batch_id} ended with status: {batch.status} and no output file.")

        # Stall detection: track whether completed count has moved
        if counts.completed != last_completed:
            last_completed = counts.completed
            stall_since = time.time()
        elif stall_since is None:
            stall_since = time.time()
        else:
            stalled_for = time.time() - stall_since
            remaining = counts.total - counts.completed
            if stalled_for >= STALL_TIMEOUT and remaining > 0:
                print(f"    Stall detected: {remaining} requests stuck for {stalled_for/60:.0f} min. Cancelling batch...")
                client.batches.cancel(batch_id)
                # Wait briefly for cancellation to propagate and output file to appear
                time.sleep(10)
                batch = client.batches.retrieve(batch_id)
                if batch.output_file_id:
                    print(f"    Cancelled. Partial output available — {remaining} rows will fall back to original text.")
                    return batch.output_file_id
                print(f"    Cancelled with no output file. All {counts.total} rows will fall back to original text.")
                return None

        time.sleep(POLL_INTERVAL)


def download_and_parse_chunk(client: OpenAI, output_file_id: str | None, prompt_num: int, chunk_idx: int) -> dict[str, str]:
    """Download result JSONL and parse into {custom_id: normalized_text}."""
    if output_file_id is None:
        print(f"    No output file for chunk {chunk_idx} — skipping download, affected rows will use original text.")
        return {}
    raw_path = WORK_DIR / f"batch_output_p{prompt_num}_chunk{chunk_idx}.jsonl"
    print(f"    Downloading results...")
    content = client.files.content(output_file_id)
    raw_path.write_bytes(content.read())

    results = {}
    failures = 0
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cid = record["custom_id"]
            if record.get("error"):
                print(f"    WARNING: failed request {cid}: {record['error']}")
                failures += 1
                continue
            text = record["response"]["body"]["choices"][0]["message"]["content"].strip()
            results[cid] = text

    print(f"    Parsed {len(results)} results, {failures} failures")
    return results


def write_output_csvs(dfs: dict, results: dict[str, str], prompt_num: int):
    for split, df in dfs.items():
        out_rows = []
        missing = 0
        for _, row in df.iterrows():
            cid = f"{split}_{row['train_id']}"
            normalized = results.get(cid)
            if normalized is None:
                print(f"  WARNING: no result for {cid}, keeping original")
                normalized = str(row.get("item_description", ""))
                missing += 1
            out_rows.append({
                "train_id":         row["train_id"],
                "name":             row.get("name", ""),
                "category_name":    row.get("category_name", ""),
                "price":            row.get("price", ""),
                "item_description": normalized,
            })

        out_df = pd.DataFrame(out_rows)
        out_path = OUTPUT_DIR / f"{split}_p{prompt_num}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  Wrote {len(out_df)} rows → {out_path}  (missing: {missing})")


def load_cached_chunk(prompt_num: int, chunk_idx: int) -> dict[str, str] | None:
    """Return cached results if this chunk was already completed in a previous run."""
    raw_path = WORK_DIR / f"batch_output_p{prompt_num}_chunk{chunk_idx}.jsonl"
    if not raw_path.exists():
        return None
    print(f"    Chunk {chunk_idx} already downloaded — loading from cache.")
    results = {}
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            cid = record["custom_id"]
            if record.get("error"):
                continue
            text = record["response"]["body"]["choices"][0]["message"]["content"].strip()
            results[cid] = text
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    WORK_DIR.mkdir(exist_ok=True)

    client = OpenAI(api_key="")  # ← paste your key here

    print("\n=== Loading splits ===")
    dfs = load_all_splits()

    for p in [1, 2]:
        print(f"\n{'='*60}")
        print(f"  PROMPT {p}")
        print(f"{'='*60}")

        print(f"\n  Building records...")
        records = build_all_records(dfs, p)
        n_chunks = math.ceil(len(records) / BATCH_CHUNK_SIZE)
        print(f"  {len(records)} total requests → {n_chunks} chunk(s) of up to {BATCH_CHUNK_SIZE}")

        all_results: dict[str, str] = {}

        for i in range(n_chunks):
            print(f"\n  --- Chunk {i+1}/{n_chunks} ---")

            # Skip if already completed in a previous run
            cached = load_cached_chunk(p, i)
            if cached is not None:
                all_results.update(cached)
                print(f"    Skipped (cached): {len(cached)} results loaded")
                continue

            # Write JSONL for this chunk
            chunk = records[i * BATCH_CHUNK_SIZE : (i + 1) * BATCH_CHUNK_SIZE]
            jsonl_path = write_chunk_jsonl(chunk, p, i)
            print(f"    Written: {jsonl_path.name} ({len(chunk)} requests)")

            # Submit, poll, download — sequentially
            batch_id = submit_chunk(client, jsonl_path, p, i)
            output_file_id = poll_until_done(client, batch_id, label=f"p{p} chunk {i+1}/{n_chunks}")
            chunk_results = download_and_parse_chunk(client, output_file_id, p, i)
            all_results.update(chunk_results)

        print(f"\n  Writing output CSVs for prompt {p}...")
        write_output_csvs(dfs, all_results, p)

    print("\n=== All done ===")
    print("Output files:")
    for p in [1, 2]:
        for split in SPLITS:
            print(f"  {split}_p{p}.csv")


if __name__ == "__main__":
    main()