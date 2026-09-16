"""Phase 4: Train the CORN-2 binary image-quality classifier (CDTI Q_image).

Classifies raw IVCCM captures as high (1) or low (0) quality, so the CDTI
can discount trust for predictions made on unusable images.

Trains on CORN-2 (train/high-quality + train/low-quality) and evaluates on
CORN-2 test sets.

Usage:
    python scripts/train_quality.py                            # configs/default.yaml
    python scripts/train_quality.py --epochs 10 --batch-size 32
    python scripts/train_quality.py --resume outputs/checkpoints/quality_corn2.pt
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.optim as optim

from src.utils.config import load_config
from src.utils.training import (
    set_seed, get_device, save_checkpoint, load_checkpoint, make_cosine_scheduler,
)
from src.data.datamodule import CornealDataModule
from src.models import build_quality_net, ClassificationMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CORNEAL-TRUST image-quality classifier")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit training batches (debug)")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit test batches (debug)")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size (debug)")
    parser.add_argument("--device", type=str, default=None, help="force device (cpu/cuda)")
    parser.add_argument("--out", type=str, default=None, help="checkpoint output path")
    parser.add_argument("--resume", type=str, default=None, help="checkpoint path to resume from")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, limit: int | None = None) -> dict[str, float]:
    model.eval()
    metrics = ClassificationMetrics(num_classes=model.num_classes)
    total_loss = 0.0
    n = 0
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        x = batch["image"].to(device)
        y = batch["label"].long().to(device)
        logits = model(x)
        total_loss += torch.nn.functional.cross_entropy(logits, y).item()
        metrics.update(logits, y)
        n += 1
    res = metrics.results()
    res["loss"] = total_loss / max(n, 1)
    model.train()
    return res


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.get("project", {}).get("seed", 42))
    device = torch.device(args.device) if args.device else get_device()
    print(f"Device: {device}")

    q_cfg = cfg.setdefault("quality", {})
    if args.epochs:
        q_cfg["epochs"] = args.epochs
    if args.batch_size:
        q_cfg["batch_size"] = args.batch_size
    if args.lr:
        q_cfg["learning_rate"] = args.lr
    if args.num_workers is not None:
        cfg.setdefault("training", {})["num_workers"] = args.num_workers

    if args.image_size:
        cfg.setdefault("data", {})["image_size"] = args.image_size

    dm = CornealDataModule(cfg)
    batch_size = q_cfg.get("batch_size", 32)
    train_loader = dm.get_train_loader("corn2", batch_size=batch_size)
    test_loader = dm.get_test_loader("corn2", batch_size=batch_size)

    model = build_quality_net(cfg).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=q_cfg.get("learning_rate", 0.001),
        weight_decay=q_cfg.get("weight_decay", 0.0001),
    )
    epochs = int(q_cfg.get("epochs", 50))
    steps_per_epoch = len(train_loader)
    scheduler = make_cosine_scheduler(optimizer, total_steps=epochs * steps_per_epoch)

    out_path = Path(args.out) if args.out else (
        ROOT / cfg.get("output", {}).get("checkpoint_dir", "outputs/checkpoints")
        / "quality_corn2.pt"
    )

    start_epoch = 1
    best_acc = 0.0
    if args.resume:
        state = load_checkpoint(args.resume, map_location=str(device))
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict"):
            scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = int(state.get("epoch", 0)) + 1
        best_acc = float(state.get("best_acc", 0.0))
        print(f"Resumed from {args.resume} (epoch {state.get('epoch')}, best_acc {best_acc:.4f})")

    print(
        f"Epochs: {epochs} | Train batches: {steps_per_epoch} | "
        f"Test batches: {len(test_loader)} | Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    patience = int(q_cfg.get("early_stopping_patience", 10))
    bad_epochs = 0
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0
        running_n = 0
        start = time.time()
        for i, batch in enumerate(train_loader):
            if args.limit and i >= args.limit:
                break
            x = batch["image"].to(device)
            y = batch["label"].long().to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_n += 1

        avg_loss = running_loss / max(running_n, 1)
        test_metrics = evaluate(model, test_loader, device, limit=args.limit_val)
        elapsed = time.time() - start
        print(
            f"[{epoch:3d}/{epochs}] loss={avg_loss:.4f} "
            f"test_loss={test_metrics['loss']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"test_f1={test_metrics['macro_f1']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if test_metrics["accuracy"] >= best_acc:
            best_acc = test_metrics["accuracy"]
            bad_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_acc": best_acc,
                    "config": cfg,
                },
                out_path,
            )
            print(f"  -> checkpoint saved (test_acc={best_acc:.4f})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement).")
                break

    print(f"\nDone. Best test accuracy: {best_acc:.4f}", flush=True)
    print(f"Checkpoint: {out_path}", flush=True)


if __name__ == "__main__":
    main()