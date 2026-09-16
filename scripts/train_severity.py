"""Phase 3: Train the 4-class corneal neuropathy severity classifier.

Trains on CORN1500 (1500 images) with CORN-3 held out for validation.

Usage:
    python scripts/train_severity.py                    # uses configs/default.yaml
    python scripts/train_severity.py --epochs 10 --batch-size 32 --limit 100
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
from src.utils.training import set_seed, get_device, save_checkpoint, make_cosine_scheduler
from src.data.datamodule import CornealDataModule
from src.models import build_severity_net, ClassificationMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CORNEAL-TRUST severity classifier")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--limit", type=int, default=None, help="Limit training batches (debug)")
    parser.add_argument("--limit-val", type=int, default=None, help="Limit val batches (debug)")
    parser.add_argument("--image-size", type=int, default=None, help="Override image size (debug)")
    parser.add_argument("--device", type=str, default=None, help="force device (cpu/cuda)")
    parser.add_argument("--out", type=str, default=None, help="checkpoint output path")
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

    sev_cfg = cfg.setdefault("severity", {})
    if args.epochs:
        sev_cfg["epochs"] = args.epochs
    if args.batch_size:
        sev_cfg["batch_size"] = args.batch_size
    if args.lr:
        sev_cfg["learning_rate"] = args.lr
    if args.num_workers is not None:
        cfg.setdefault("training", {})["num_workers"] = args.num_workers

    if args.image_size:
        cfg.setdefault("data", {})["image_size"] = args.image_size

    dm = CornealDataModule(cfg)
    train_loader = dm.get_combined_severity_loader(
        "train", batch_size=sev_cfg.get("batch_size", 32)
    )
    val_loader = dm.get_combined_severity_loader(
        "val", batch_size=sev_cfg.get("batch_size", 32)
    )

    model = build_severity_net(cfg).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=sev_cfg.get("learning_rate", 0.001),
        weight_decay=sev_cfg.get("weight_decay", 0.0001),
    )
    epochs = int(sev_cfg.get("epochs", 50))
    steps_per_epoch = len(train_loader)
    scheduler = make_cosine_scheduler(optimizer, total_steps=epochs * steps_per_epoch)

    out_path = Path(args.out) if args.out else (
        ROOT / cfg.get("output", {}).get("checkpoint_dir", "outputs/checkpoints")
        / "severity_corn1500.pt"
    )

    print(
        f"Epochs: {epochs} | Train batches: {steps_per_epoch} | "
        f"Val batches: {len(val_loader)} | Params: {sum(p.numel() for p in model.parameters()):,}"
    )

    best_acc = 0.0
    patience = int(sev_cfg.get("early_stopping_patience", 10))
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
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
        val_metrics = evaluate(model, val_loader, device, limit=args.limit_val)
        elapsed = time.time() - start
        print(
            f"[{epoch:3d}/{epochs}] loss={avg_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        if val_metrics["accuracy"] >= best_acc:
            best_acc = val_metrics["accuracy"]
            bad_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_acc": best_acc,
                    "config": cfg,
                },
                out_path,
            )
            print(f"  -> checkpoint saved (val_acc={best_acc:.4f})", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement).")
                break

    print(f"\nDone. Best validation accuracy: {best_acc:.4f}", flush=True)
    print(f"Checkpoint: {out_path}", flush=True)


if __name__ == "__main__":
    main()