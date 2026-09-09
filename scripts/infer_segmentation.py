"""Phase 2: Run segmentation inference with a trained checkpoint.

Usage:
    python scripts/infer_segmentation.py --checkpoint outputs/checkpoints/unet_corn1.pt
    python scripts/infer_segmentation.py --image ../CORN-1/images/train_orig/xxx.tif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
import tifffile

from src.utils.config import load_config
from src.utils.training import get_device, load_checkpoint
from src.models import build_unet
from src.utils.skeleton import skeletonize_binary, topological_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORNEAL-TRUST segmentation inference")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Single image to segment")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default=None, help="Save prediction masks here")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def predict(model, image: np.ndarray, device, threshold: float) -> np.ndarray:
    """Run segmentation on a single (H,W) grayscale image."""
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    x = x / 255.0
    logits = model(x)
    mask = (torch.sigmoid(logits).squeeze() > threshold).cpu().numpy()
    return mask


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    model = build_unet(cfg)
    state = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint (epoch {state.get('epoch')}, best_dice {state.get('best_dice'):.4f})")

    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    def process(path: Path):
        img = tifffile.imread(str(path))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = predict(model, img, device, args.threshold)
        skel = skeletonize_binary(mask)
        topo = topological_features(skel)
        print(f"\n{path.name}")
        print(f"  predicted mask area: {mask.mean()*100:.2f}%")
        print(f"  junctions={topo['n_junctions']} endpoints={topo['endpoints']} "
              f"length={topo['total_length']:.0f} px")
        if args.out_dir:
            out = Path(args.out_dir) / (path.stem + "_pred.png")
            cv2.imwrite(str(out), (mask * 255).astype(np.uint8))
            print(f"  saved -> {out}")
        return mask

    if args.image:
        process(Path(args.image))
    elif args.image_dir:
        for p in sorted(Path(args.image_dir).glob("*.tif")):
            process(p)
    else:
        print("Provide --image or --image-dir.")


if __name__ == "__main__":
    main()
