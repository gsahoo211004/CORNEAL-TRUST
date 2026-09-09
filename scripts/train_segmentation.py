"""Phase 2: Train the corneal nerve segmentation U-Net (CORN-1).

Usage:
    python scripts/train_segmentation.py                # uses configs/default.yaml
    python scripts/train_segmentation.py --config my.yaml
    python scripts/train_segmentation.py --epochs 10 --batch-size 4 --limit 100
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
from src.models import build_unet, AMCLLoss, SegmentationMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CORNEAL-TRUST segmentation U-Net")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--limit", type=int, default=None, help="Limit training batches (debug)")
    parser.add_argument("--device", type=str, default=None, help="force device (cpu/cuda)")
    parser.add_argument("--out", type=str, default=None, help="checkpoint output path")
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict[str, float]:
    """Evaluate the model on a validation loader."""
    model.eval()
    metrics = SegmentationMetrics()
    total_loss = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        pred = model(x)
        total_loss += loss_fn(pred, y).item()
        metrics.update(pred, y)
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

    # Apply CLI overrides
    train_cfg = cfg.setdefault("training", {})
    if args.epochs:
        train_cfg["epochs"] = args.epochs
    if args.batch_size:
        train_cfg["batch_size"] = args.batch_size
    if args.lr:
        train_cfg["learning_rate"] = args.lr

    dm = CornealDataModule(cfg)
    train_loader = dm.get_train_loader("corn1")
    val_loader = dm.get_val_loader("corn1")

    model = build_unet(cfg).to(device)
    loss_fn = AMCLLoss(
        alpha=cfg.get("model", {}).get("loss_alpha", 1.0),
        beta=cfg.get("model", {}).get("loss_beta", 0.5),
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_cfg.get("learning_rate", 0.001),
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    epochs = int(train_cfg.get("epochs", 50))
    steps_per_epoch = len(train_loader)
    scheduler = make_cosine_scheduler(optimizer, total_steps=epochs * steps_per_epoch)

    out_path = Path(args.out) if args.out else (
        ROOT / cfg.get("output", {}).get("checkpoint_dir", "outputs/checkpoints") / "unet_corn1.pt"
    )

    print(f"Epochs: {epochs} | Samples/epoch: {steps_per_epoch} | Params: {sum(p.numel() for p in model.parameters()):,}")

    best_dice = 0.0
    patience = int(train_cfg.get("early_stopping_patience", 10))
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
            y = batch["mask"].to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            running_n += 1

        avg_loss = running_loss / max(running_n, 1)
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        elapsed = time.time() - start
        print(
            f"[{epoch:3d}/{epochs}] loss={avg_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"dice={val_metrics['dice']:.4f} iou={val_metrics['iou']:.4f} "
            f"acc={val_metrics['accuracy']:.4f} ({elapsed:.1f}s)"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            bad_epochs = 0
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                    "config": cfg,
                },
                out_path,
            )
            print(f"  -> checkpoint saved (dice={best_dice:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping after {epoch} epochs (no improvement).")
                break

    print(f"\nDone. Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoint: {out_path}")


if __name__ == "__main__":
    main()
