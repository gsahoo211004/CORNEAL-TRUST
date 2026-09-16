"""Phase 4: Unified trust evaluation on CORN-3 + CORN-2-test.

Loads the trained softmax severity, evidential severity, and CORN-2
quality checkpoints; evaluates severity/evidential on the CORN-3
validation set and quality on the CORN-2 test set; computes CDTI +
referral analysis (LowTrust rate, recall-safety, coverage-vs-accuracy).

Usage:
    python scripts/evaluate_trust.py
    python scripts/evaluate_trust.py --limit 100 --image-size 192
    python scripts/evaluate_trust.py --evidential-checkpoint PATH --quality-checkpoint PATH
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from src.utils.config import load_config
from src.utils.training import get_device, load_checkpoint
from src.data.datamodule import CornealDataModule
from src.models import (
    build_severity_net, build_evidential_net, build_quality_net,
    evidential_predict, ece_score, trust_report,
)

GRADE_NAMES = ["grade1", "grade2", "grade3", "grade4"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CORNEAL-TRUST unified trust evaluation")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="outputs/checkpoints/severity_corn1500.pt")
    parser.add_argument("--evidential-checkpoint", type=str, default="outputs/checkpoints/severity_corn1500_evidential.pt")
    parser.add_argument("--quality-checkpoint", type=str, default="outputs/checkpoints/quality_corn2.pt")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=None, help="override image size (smoke)")
    parser.add_argument("--limit", type=int, default=None, help="limit val batches (smoke)")
    parser.add_argument("--out-dir", type=str, default="outputs/results")
    return parser.parse_args()


def accuracy_macro_f1(pred: np.ndarray, true: np.ndarray, k: int) -> tuple[float, float]:
    acc = float(np.mean(pred == true))
    f1s = []
    for c in range(k):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1s.append(2 * prec * rec / (prec + rec + 1e-6))
    return acc, float(np.mean(f1s))


def confusion_matrix(pred: np.ndarray, true: np.ndarray, k: int) -> np.ndarray:
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(true.tolist(), pred.tolist()):
        cm[t, p] += 1
    return cm


def coverage_curve(rows: list[dict], n: int = 41) -> list[dict]:
    rows = sorted(rows, key=lambda r: r["cdti"], reverse=True)
    out = []
    for i in range(n):
        t = i / (n - 1)
        sel = [r for r in rows if r["cdti"] >= t]
        if not sel:
            out.append({"threshold": round(t, 3), "coverage": 0.0, "accuracy": 0.0})
            continue
        acc = float(np.mean([r["pred"] == r["true"] for r in sel]))
        out.append({"threshold": round(t, 3), "coverage": round(len(sel) / len(rows), 4), "accuracy": round(acc, 4)})
    return out


@torch.no_grad()
def eval_severity(model, loader, device, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    model.eval()
    preds, trues, confs = [], [], []
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        x = batch["image"].to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        preds.extend(logits.argmax(dim=1).tolist())
        trues.extend(batch["label"].long().tolist())
        confs.extend(probs.max(dim=1).values.tolist())
    model.train()
    return np.array(preds), np.array(trues), confs


@torch.no_grad()
def eval_evidential(
    model, quality_model, loader, device, cfg, limit: int | None = None
) -> tuple[np.ndarray, np.ndarray, list[dict], np.ndarray]:
    model.eval()
    qm_eval = quality_model.eval() if quality_model is not None else None
    preds, trues = [], []
    all_probs: list[torch.Tensor] = []
    rows: list[dict] = []
    is_c3 = True  # severity loader rows carry true grade labels
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        x = batch["image"].to(device)
        evidence = model(x)
        out = evidential_predict(evidence)
        probs = out["probs"]
        all_probs.append(probs.cpu())
        paths = batch.get("image_path", [f"batch{i}"] * x.shape[0])
        q = torch.ones(x.shape[0])
        if quality_model is not None:
            q = torch.softmax(quality_model(x), dim=1)[:, 1].cpu()
        for j in range(x.shape[0]):
            report = trust_report(
                confidence=float(out["confidence"][j].item()),
                uncertainty=float(out["uncertainty"][j].item()),
                q_image=float(q[j].item()),
                cfg=cfg,
            )
            rows.append({
                "path": paths[j],
                "true": int(batch["label"][j].item()) if is_c3 else -1,
                "pred": int(out["pred"][j].item()),
                "conf": float(out["confidence"][j].item()),
                "unc": float(out["uncertainty"][j].item()),
                "q": float(q[j].item()),
                "cdti": report["cdti"],
                "trusted": report["trusted"],
            })
            preds.append(int(out["pred"][j].item()))
            trues.append(int(batch["label"][j].item()) if is_c3 else -1)
    model.train()
    if qm_eval is not None:
        qm_eval.train()
    ece = float(ece_score(torch.cat(all_probs), torch.tensor(trues)))
    return np.array(preds), np.array(trues), rows, ece


@torch.no_grad()
def eval_quality(model, loader, device, limit: int | None = None) -> tuple[np.ndarray, np.ndarray, list[float]]:
    model.eval()
    preds, trues, confs = [], [], []
    for i, batch in enumerate(loader):
        if limit and i >= limit:
            break
        x = batch["image"].to(device)
        probs = torch.softmax(model(x), dim=1)
        preds.extend(probs.argmax(dim=1).tolist())
        trues.extend(batch["label"].long().tolist())
        confs.extend(probs.max(dim=1).values.tolist())
    model.train()
    return np.array(preds), np.array(trues), confs


def write_csv(path: Path, header: list[str], rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.image_size:
        cfg.setdefault("data", {})["image_size"] = args.image_size
    device = torch.device(args.device) if args.device else get_device()
    dm = CornealDataModule(cfg)

    ckpt_sev = Path(args.checkpoint)
    ckpt_ev = Path(args.evidential_checkpoint)
    ckpt_q = Path(args.quality_checkpoint)

    results: dict[str, object] = {}

    # 1) Softmax severity on CORN-3
    loader = dm.get_combined_severity_loader("val", batch_size=32)
    if ckpt_sev.exists():
        model = build_severity_net(cfg).to(device)
        model.load_state_dict(load_checkpoint(ckpt_sev, map_location=str(device))["model_state_dict"])
        pred, true, conf = eval_severity(model, loader, device, limit=args.limit)
        acc, f1 = accuracy_macro_f1(pred, true, 4)
        print("[softmax severity] CORN-3")
        print(f"  accuracy={acc:.4f} | macro-F1={f1:.4f} | mean_conf={float(np.mean(conf)):.4f}")
        np.set_printoptions(linewidth=120)
        print("  confusion (row=truth, col=pred):\n", confusion_matrix(pred, true, 4))
        results["severity_accuracy"] = round(acc, 4)
        results["severity_macro_f1"] = round(f1, 4)
    else:
        print(f"!! skipping softmax severity (missing {ckpt_sev})")

    # 2) Evidential severity + CDTI on CORN-3
    if ckpt_ev.exists():
        model = build_evidential_net(cfg).to(device)
        model.load_state_dict(load_checkpoint(ckpt_ev, map_location=str(device))["model_state_dict"])
        qmodel = None
        if ckpt_q.exists():
            qmodel = build_quality_net(cfg).to(device)
            qmodel.load_state_dict(load_checkpoint(ckpt_q, map_location=str(device))["model_state_dict"])
        pred, true, rows, ece = eval_evidential(model, qmodel, loader, device, cfg, limit=args.limit)
        acc, f1 = accuracy_macro_f1(pred, true, 4)
        mean_unc = float(np.mean([r["unc"] for r in rows]))
        mean_q = float(np.mean([r["q"] for r in rows]))
        lowtrust = [r for r in rows if not r["trusted"]]
        n_lt = len(lowtrust)
        true_high_lt = sum(1 for r in lowtrust if r["true"] >= 2)
        high_total = sum(1 for r in rows if r["true"] >= 2)
        high_trusted = sum(1 for r in rows if r["true"] >= 2 and r["trusted"])
        print(f"\n[evidential severity] CORN-3 (q_model={'yes' if qmodel else 'no (Q=1.0)'})")
        print(f"  accuracy={acc:.4f} | macro-F1={f1:.4f} | ECE={ece:.4f} | mean_unc={mean_unc:.4f} | mean_Q={mean_q:.4f}")
        threshold = cfg.get("trust", {}).get("referral_threshold", 0.60)
        print(f"  LowTrust (< {threshold}): {n_lt}/{len(rows)} ({100*n_lt/max(len(rows),1):.1f}%)")
        print(f"  recall-safety: of {n_lt} LowTrust, {true_high_lt} are true grade3/4 ({100*true_high_lt/max(n_lt,1):.1f}%)")
        print(f"  high-grade auto-cleared (trusted but truth grade3/4): {high_trusted}/{high_total}")
        cov = coverage_curve(rows)
        cov60 = next((c for c in cov if c["threshold"] == 0.6), None)
        if cov60 and cov60["coverage"] > 0:
            print(f"  at CDTI>=0.60: coverage={cov60['coverage']*100:.1f}% | accuracy-on-accepted={cov60['accuracy']*100:.1f}%")
        print("  confusion (row=truth, col=pred):\n", confusion_matrix(pred, true, 4))
        results["evidential_accuracy"] = round(acc, 4)
        results["evidential_macro_f1"] = round(f1, 4)
        results["evidential_ece"] = round(ece, 4)
        results["mean_uncertainty"] = round(mean_unc, 4)
        results["mean_q_image"] = round(mean_q, 4)
        results["lowtrust_count"] = n_lt
        results["lowtrust_rate"] = round(n_lt / max(len(rows), 1), 4)
        results["recall_safety_lowtrust"] = round(true_high_lt / max(n_lt, 1), 4)
        results["high_grade_auto_cleared"] = high_trusted
        results["high_grade_total"] = high_total

        out_dir = ROOT / args.out_dir
        write_csv(
            out_dir / "trust_eval_predictions.csv",
            ["path", "true_label", "pred_label", "confidence", "uncertainty", "q_image", "cdti", "verdict"],
            [[r["path"], r["true"], r["pred"], round(r["conf"], 4), round(r["unc"], 4),
              round(r["q"], 4), round(r["cdti"], 4), "Trusted" if r["trusted"] else "LowTrust"] for r in rows],
        )
        write_csv(
            out_dir / "trust_eval_coverage.csv",
            ["threshold", "coverage", "accuracy"],
            [[c["threshold"], round(c["coverage"], 4), c["accuracy"]] for c in cov],
        )
    else:
        print(f"!! skipping evidential (missing {ckpt_ev})")

    # 3) Quality model on CORN-2 test
    if ckpt_q.exists():
        qmodel = build_quality_net(cfg).to(device)
        qmodel.load_state_dict(load_checkpoint(ckpt_q, map_location=str(device))["model_state_dict"])
        qpred, qtrue, qconf = eval_quality(
            qmodel, dm.get_test_loader("corn2", batch_size=32), device, limit=args.limit
        )
        qacc, qf1 = accuracy_macro_f1(qpred, qtrue, 2)
        low_recall = float(np.mean(qpred[qtrue == 0] == 0)) if np.any(qtrue == 0) else float("nan")
        print(f"\n[quality] CORN-2 test")
        print(f"  accuracy={qacc:.4f} | macro-F1={qf1:.4f} | low-quality recall={low_recall:.4f} | test_n={len(qtrue)}")
        results["quality_test_accuracy"] = round(qacc, 4)
        results["quality_test_macro_f1"] = round(qf1, 4)
        results["quality_low_recall"] = round(low_recall, 4)
    else:
        print(f"!! skipping quality (missing {ckpt_q})")

    # Summary CSV
    out_dir = ROOT / args.out_dir
    write_csv(
        out_dir / "trust_eval_metrics.csv",
        ["metric", "value"],
        [[k, v] for k, v in results.items()],
    )
    print(f"\nSaved: {out_dir}/trust_eval_{{metrics,coverage,predictions}}.csv")


if __name__ == "__main__":
    main()