"""Phase 3: Run 4-class severity inference + triage with a trained checkpoint.

Usage:
    python scripts/infer_severity.py --checkpoint outputs/checkpoints/severity_corn1500.pt
    python scripts/infer_severity.py --checkpoint ... --image ../CORN1500/level1/1_level1.jpg
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
import tifffile
import cv2

from src.utils.config import load_config
from src.utils.training import get_device, load_checkpoint
from src.models import build_severity_net
from src.knowledge_graph import TriageEngine, load_triage_engine

GRADE_NAMES = ["Level 1 (normal)", "Level 2 (mild)", "Level 3 (moderate)", "Level 4 (severe)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORNEAL-TRUST severity inference + triage")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Severity .pt checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Single image to grade")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


@torch.no_grad()
def predict(model, image: np.ndarray, device) -> tuple[torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    x = x / 255.0
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    grade = logits.argmax(dim=1).item() + 1
    return grade, probs


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".jpg":
        return np.array(Image.open(str(path)).convert("L"))
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()
    eng: TriageEngine = load_triage_engine(cfg)

    model = build_severity_net(cfg)
    state = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint (epoch {state.get('epoch')}, best_acc {state.get('best_acc'):.4f})\n")

    def process(path: Path):
        img = load_image(path)
        grade, probs = predict(model, img, device)
        triage = eng.triage(severity_grade=grade)
        print(f"\n{path.name}")
        print(f"  predicted: {GRADE_NAMES[grade - 1]} (prob {probs.max().item():.3f})")
        print(f"  risk: {triage['risk_level']} | referral window: {triage['referral_window_days']} days")
        print(f"  rules fired: {triage['structural_rules_fired']}")
        print(f"  referral: {triage['referral_text']}")
        return grade, probs

    if args.image:
        process(Path(args.image))
    elif args.image_dir:
        exts = ("*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png")
        files: list[Path] = []
        for e in exts:
            files.extend(Path(args.image_dir).glob(e))
        for p in sorted(files):
            process(p)
    else:
        print("Provide --image or --image-dir.")


if __name__ == "__main__":
    main()