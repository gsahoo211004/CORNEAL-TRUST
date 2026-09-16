"""Phase 3/4: Run 4-class severity inference + triage with a trained checkpoint.

Phase 3 (default): softmax probabilities + knowledge-graph triage.
Phase 4 (--evidential): uncertainty-aware (Dirichlet) output + Corneal
Diagnostic Trust Index (CDTI); LowTrust predictions are escalated for
manual review even if the predicted grade is low.

Usage:
    python scripts/infer_severity.py --checkpoint outputs/checkpoints/severity_corn1500.pt
    python scripts/infer_severity.py --checkpoint ... \\ --image ../CORN1500/level1/1_level1.jpg
    python scripts/infer_severity.py --checkpoint outputs/checkpoints/severity_corn1500_evidential.pt \
        --evidential --quality-checkpoint outputs/checkpoints/quality_corn2.pt
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
from src.models import (
    build_severity_net, build_evidential_net, build_quality_net,
    evidential_predict, trust_report,
)
from src.knowledge_graph import TriageEngine, load_triage_engine

GRADE_NAMES = ["Level 1 (normal)", "Level 2 (mild)", "Level 3 (moderate)", "Level 4 (severe)"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORNEAL-TRUST severity inference + triage")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True, help="Severity .pt checkpoint")
    parser.add_argument("--image", type=str, default=None, help="Single image to grade")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory of images")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--evidential", action="store_true", help="use evidential head + CDTI")
    parser.add_argument(
        "--quality-checkpoint", type=str, default=None,
        help="CORN-2 quality model for CDTI Q_image (required with --evidential for real Q)",
    )
    return parser.parse_args()


@torch.no_grad()
def predict_evidential(model, image: np.ndarray, device) -> dict[str, torch.Tensor]:
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    x = x / 255.0
    return evidential_predict(model(x))


@torch.no_grad()
def predict_softmax(model, image: np.ndarray, device) -> dict[str, torch.Tensor]:
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    x = x / 255.0
    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    return {
        "probs": probs,
        "confidence": probs.max(dim=1).values,
        "pred": probs.argmax(dim=1),
    }


@torch.no_grad()
def predict_quality(model, image: np.ndarray, device) -> float:
    """Return prob(class=1 = high quality) from the CORN-2 model."""
    x = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    x = x / 255.0
    return float(torch.softmax(model(x), dim=1)[0, 1].item())


def load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".jpg":
        return np.array(Image.open(str(path)).convert("L"))
    img = tifffile.imread(str(path))
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img


def apply_trust_override(triage: dict, verdict: dict) -> dict:
    """Escalate triage when CDTI declares the prediction LowTrust."""
    if verdict["trusted"]:
        triage["trust"] = verdict
        return triage
    triage["risk_level"] = "High"
    triage["referral_window_days"] = min(triage["referral_window_days"], 7)
    note = (
        f"Low system trust (CDTI {verdict['cdti']:.3f} < {verdict['threshold']:.2f}) — "
        "the image could not be graded with sufficient confidence. Escalate for manual review."
    )
    triage["referral_text"] = note + " " + triage["referral_text"]
    triage["trust"] = verdict
    triage["low_trust_escalation"] = True
    return triage


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()
    eng: TriageEngine = load_triage_engine(cfg)

    if args.evidential:
        model = build_evidential_net(cfg)
        mode = "evidential"
    else:
        model = build_severity_net(cfg)
        mode = "softmax"

    state = load_checkpoint(args.checkpoint, map_location=str(device))
    model.load_state_dict(state["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {mode} checkpoint (epoch {state.get('epoch')}, best_acc {state.get('best_acc'):.4f})")

    quality_model = None
    if args.quality_checkpoint:
        quality_model = build_quality_net(cfg)
        q_state = load_checkpoint(args.quality_checkpoint, map_location=str(device))
        quality_model.load_state_dict(q_state["model_state_dict"])
        quality_model.to(device).eval()
        print(f"Loaded quality checkpoint (epoch {q_state.get('epoch')}, best_acc {q_state.get('best_acc'):.4f})")

    def process(path: Path):
        img = load_image(path)
        if args.evidential:
            out = predict_evidential(model, img, device)
            grade = int(out["pred"].item()) + 1
            probs = out["probs"]
            confidence = float(out["confidence"].item())
            uncertainty = float(out["uncertainty"].item())
        else:
            out = predict_softmax(model, img, device)
            grade = int(out["pred"].item()) + 1
            probs = out["probs"]
            confidence = float(out["confidence"].item())
            uncertainty = None

        q_image = 1.0
        if quality_model is not None:
            q_image = predict_quality(quality_model, img, device)

        triage = eng.triage(severity_grade=grade)
        trust_line = ""
        if args.evidential:
            verdict = trust_report(
                confidence=confidence, uncertainty=uncertainty, q_image=q_image, cfg=cfg
            )
            triage = apply_trust_override(triage, verdict)
            trust_line = (
                f"\n  uncertainty={uncertainty:.3f} | Q_image={q_image:.3f} | "
                f"CDTI={verdict['cdti']:.3f} | verdict={verdict['verdict']}"
            )

        print(f"\n{path.name}")
        print(f"  predicted: {GRADE_NAMES[grade - 1]} (prob {confidence:.3f})")
        if trust_line:
            print(trust_line)
        print(f"  risk: {triage['risk_level']} | referral window: {triage['referral_window_days']} days")
        print(f"  rules fired: {triage['structural_rules_fired']}")
        print(f"  referral: {triage['referral_text']}")
        return grade, triage

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