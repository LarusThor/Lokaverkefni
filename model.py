# src/model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel


@dataclass
class ModelOutput:
    preds: torch.Tensor
    loss: Optional[torch.Tensor]


class PriceRegressor(nn.Module):
    """
    Fine-tunes transformer + regression head.
    Input: tokenized text (name + description)
    Output: log1p(price)
    """
    def __init__(
        self,
        encoder_name: str,
        dropout: float = 0.1,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        h = self.encoder.config.hidden_size

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(h, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None) -> ModelOutput:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]          # [B, H]
        pred = self.head(cls).squeeze(-1)             # [B]

        loss = None
        if labels is not None:
            loss = self.loss_fn(pred, labels)

        return ModelOutput(preds=pred, loss=loss)