# src/evaluate.py
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import torch


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device) -> Dict[str, float]:
    """
    Requires labels to exist in loader batches.
    Computes RMSE/MAE on original price scale by expm1.
    """
    model.eval()

    preds_log, labels_log = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)

        preds_log.append(out.preds.detach().cpu().numpy())
        labels_log.append(labels.detach().cpu().numpy())

    preds_log = np.concatenate(preds_log)
    labels_log = np.concatenate(labels_log)

    preds = np.expm1(preds_log)
    labels = np.expm1(labels_log)

    rmse = float(np.sqrt(np.mean((preds - labels) ** 2)))
    mae = float(np.mean(np.abs(preds - labels)))
    return {"rmse": rmse, "mae": mae}


@torch.no_grad()
def predict_loader(model, loader, device: torch.device) -> np.ndarray:
    """
    Works even if loader has no labels (e.g., your test file).
    Returns predictions on original price scale.
    """
    model.eval()
    preds_log = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        preds_log.append(out.preds.detach().cpu().numpy())

    preds_log = np.concatenate(preds_log)
    return np.expm1(preds_log)


def delta_stats(pred_prof: np.ndarray, pred_orig: np.ndarray) -> Dict[str, float]:
    delta = pred_prof - pred_orig
    return {
        "delta_mean": float(delta.mean()),
        "delta_median": float(np.median(delta)),
        "delta_std": float(delta.std()),
        "pct_positive": float((delta > 0).mean()),
        "pct_negative": float((delta < 0).mean()),
    }