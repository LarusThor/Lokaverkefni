"""
Normalization Pilot Script
Tests both prompts on a small sample (default 30 rows) using live API calls
(not Batch API) so you get results immediately for manual inspection.

Usage:
    python normalize_pilot.py --input train.tsv --n 30 --output pilot_output.tsv
"""

import os
import argparse
import pandas as pd
from openai import OpenAI

MODEL = "gpt-4o-mini"

PROMPTS = {
    1: (
        "You are a text normalizer for a consumer marketplace price prediction system. "
        "Your task is to rewrite product listings into clear, fluent prose descriptions. "
        "Expand all abbreviations and acronyms. Correct spelling errors. Normalize condition descriptions "
        "(e.g. 'gr8 cond' → 'great condition'). Improve sentence structure and clarity. "
        "Interpret and clarify vague or non-standard product descriptions using context "
        "(e.g. 'skin energy drink' → 'facial moisturizer'). "
        "Remove irrelevant seller noise such as shipping policy, payment warnings, follower requests, "
        "and self-promotional text — keep only information relevant to the product itself. "
        "IMPORTANT: Preserve all factual details exactly — brand names, model numbers, storage sizes, "
        "colors, and specifications must not be altered, invented, or removed. "
        "Return only the rewritten listing as plain prose, no bullet points or headers."
    ),
    2: (
        "You are a text normalizer for price prediction. Rewrite the product listing as clean, "
        "plain prose: expand abbreviations, fix spelling, normalize informal language, interpret vague "
        "or creative product descriptions into standard terminology, and remove irrelevant seller text "
        "such as shipping notes or payment warnings. "
        "Do not change or invent any factual details. Return only the rewritten text, no bullet points."
    ),
}


def build_text(row: pd.Series) -> str:
    parts = []
    if pd.notna(row.get("category_name")) and str(row["category_name"]).strip():
        parts.append(f"Category: {row['category_name']}.")
    if pd.notna(row.get("name")) and str(row["name"]).strip():
        parts.append(str(row["name"]))
    if pd.notna(row.get("item_description")) and str(row["item_description"]).strip():
        parts.append(str(row["item_description"]))
    return " ".join(parts).strip()


def normalize_single(client: OpenAI, text: str, prompt_text: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=300,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user",   "content": text},
        ],
    )
    return response.choices[0].message.content.strip()


def run_pilot(input_path: str, output_path: str, n: int):
    client = OpenAI(api_key="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")

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    # Sample n rows, trying to get variety across categories
    if "category_name" in df.columns:
        sample = (
            df.groupby("category_name", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), max(1, n // df["category_name"].nunique()))))
            .head(n)
            .reset_index(drop=True)
        )
    else:
        sample = df.sample(n=min(n, len(df)), random_state=42).reset_index(drop=True)

    print(f"Sampled {len(sample)} rows across {sample['category_name'].nunique() if 'category_name' in sample.columns else 'N/A'} categories.\n")

    results = []
    for idx, row in sample.iterrows():
        original_text = build_text(row)
        print(f"[{idx + 1}/{len(sample)}] Processing...")
        print(f"  ORIGINAL : {original_text[:120]}...")

        normalized_p1 = normalize_single(client, original_text, PROMPTS[1])
        normalized_p2 = normalize_single(client, original_text, PROMPTS[2])

        print(f"  PROMPT 1 : {normalized_p1[:120]}...")
        print(f"  PROMPT 2 : {normalized_p2[:120]}...")
        print()

        results.append({
            "train_id":        row.get("train_id", idx),
            "category_name":   row.get("category_name", ""),
            "price":           row.get("price", ""),
            "original_text":   original_text,
            "normalized_p1":   normalized_p1,
            "normalized_p2":   normalized_p2,
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, sep="\t", index=False)
    print(f"Pilot output saved to {output_path}")
    print(f"\n--- Summary ---")
    print(f"Rows processed : {len(output_df)}")
    print(f"Avg original length  : {output_df['original_text'].str.len().mean():.0f} chars")
    print(f"Avg normalized_p1 len: {output_df['normalized_p1'].str.len().mean():.0f} chars")
    print(f"Avg normalized_p2 len: {output_df['normalized_p2'].str.len().mean():.0f} chars")
    print(f"\nOpen {output_path} to manually inspect side-by-side outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pilot normalization on a small sample")
    parser.add_argument("--input",  required=True, help="Input TSV file (e.g. train.tsv)")
    parser.add_argument("--output", default="pilot_output.tsv", help="Output TSV file")
    parser.add_argument("--n",      type=int, default=30, help="Number of rows to sample")
    args = parser.parse_args()

    run_pilot(args.input, args.output, args.n)