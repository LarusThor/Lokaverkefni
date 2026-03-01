# src/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_text(name: str, desc: str) -> str:
    name = "" if pd.isna(name) else str(name)
    desc = "" if pd.isna(desc) else str(desc)
    # keep a separator token-ish marker; the tokenizer will handle it as normal text
    return f"{name} [SEP] {desc}".strip()


class MercariTextDataset(Dataset):
    """
    Uses only:
      - name
      - item_description
      - price (optional; present for train/val)
    Produces tokenized tensors + optional labels (log1p(price)).
    """
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        max_length: int = 128,
        has_labels: bool = True,
        price_col: str = "price",
    ):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.has_labels = has_labels
        self.price_col = price_col

        # Prebuild texts once (faster + deterministic)
        self.texts = [
            build_text(n, d)
            for n, d in zip(self.df["name"], self.df["item_description"])
        ]

        if self.has_labels:
            if self.price_col not in self.df.columns:
                raise ValueError(f"has_labels=True but '{self.price_col}' not in df columns")
            # log1p is standard for price regression
            self.labels = np.log1p(self.df[self.price_col].astype(float).values)
        else:
            self.labels = None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        text = self.texts[idx]

        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        item = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
        }

        if self.has_labels:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float32)

        return item