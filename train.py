# src/train.py
from __future__ import annotations

import os
import json
from dataclasses import dataclass, asdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from dataset import MercariTextDataset
from model import PriceRegressor
from evaluate import evaluate_loader


@dataclass
class TrainConfig:
    encoder_name: str = "distilbert-base-uncased"
    max_length: int = 128

    lr: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 16
    num_epochs: int = 3
    warmup_ratio: float = 0.06
    grad_clip: float = 1.0

    num_workers: int = 2
    seed: int = 42
    out_dir: str = "checkpoints/price_model"


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_from_csv(train_csv: str, val_csv: str, cfg: TrainConfig):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)

    # --- minimal cleaning ---

    # Keep only necessary columns
    cols = ["train_id", "name", "item_description", "price"]
    train_df = train_df[cols]
    val_df   = val_df[cols]

    # Remove non-positive prices (recommended)
    train_df = train_df[train_df["price"] > 0].reset_index(drop=True)
    val_df   = val_df[val_df["price"] > 0].reset_index(drop=True)

    # Replace placeholder descriptions
    train_df["item_description"] = train_df["item_description"].replace("No description yet", "")
    val_df["item_description"]   = val_df["item_description"].replace("No description yet", "")

    train_ds = MercariTextDataset(train_df, tokenizer, max_length=cfg.max_length, has_labels=True)
    val_ds = MercariTextDataset(val_df, tokenizer, max_length=cfg.max_length, has_labels=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=(device.type == "cuda")
    )

    model = PriceRegressor(cfg.encoder_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    total_steps = cfg.num_epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    with open(os.path.join(cfg.out_dir, "train_config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    best_val_rmse = float("inf")

    # Track metrics each epoch for plotting (written to CSV for Colab-friendly analysis)
    history = []
    history_path = os.path.join(cfg.out_dir, "history.csv")

    for epoch in range(1, cfg.num_epochs + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.num_epochs}", leave=False)
        for batch_idx, batch in enumerate(pbar, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item())

            # Live training feedback in progress bar (loss is MSE on log1p scale)
            avg_so_far = running / batch_idx
            pbar.set_postfix(train_mse=f"{avg_so_far:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = running / max(1, len(train_loader))
        val_metrics = evaluate_loader(model, val_loader, device=device)
        val_rmse = val_metrics["rmse"]

        # Log metrics + LR each epoch (train_mse is on log1p scale; val metrics on original price scale)
        current_lr = optimizer.param_groups[0]["lr"]
        history.append({
            "epoch": epoch,
            "train_mse": avg_loss,
            "val_rmse": val_metrics["rmse"],
            "val_mae": val_metrics["mae"],
            "lr": current_lr,
        })
        pd.DataFrame(history).to_csv(history_path, index=False)

        print(
            f"Epoch {epoch}/{cfg.num_epochs} | "
            f"train_mse={avg_loss:.4f} | val_rmse={val_rmse:.4f} | val_mae={val_metrics['mae']:.4f}"
        )

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            ckpt_path = os.path.join(cfg.out_dir, "best.pt")
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "encoder_name": cfg.encoder_name,
                    "best_val_rmse": best_val_rmse,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  ✅ saved best -> {ckpt_path}")

    print(f"Done. Best val RMSE: {best_val_rmse:.4f}")


if __name__ == "__main__":
    # Example paths (your local/colab paths may differ)
    train_from_csv("train_sample.csv", "validation_sample.csv", TrainConfig())