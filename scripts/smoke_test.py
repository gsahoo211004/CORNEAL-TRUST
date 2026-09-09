"""Quick smoke test to verify all datasets load correctly."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.datamodule import CornealDataModule


def main():
    cfg = load_config()
    dm = CornealDataModule(cfg)

    print("=" * 60)
    print("CORNEAL-TRUST — Dataset Smoke Test")
    print("=" * 60)

    summary = dm.summary()
    for ds_name, counts in summary.items():
        print(f"\n{ds_name}:")
        if counts.get("exists") == 0:
            print("  NOT FOUND")
            continue
        for split, n in counts.items():
            print(f"  {split}: {n} samples")

    print("\n" + "=" * 60)
    print("Loading batches...")
    print("=" * 60)

    for ds_name in ["corn1", "corn2", "corn3", "corn1500"]:
        root = ROOT.parent / cfg.get(ds_name, {}).get("root", "").lstrip("../")
        if not root.exists():
            print(f"\n{ds_name}: SKIPPED (data not found at {root})")
            continue

        try:
            if ds_name == "corn1":
                loader = dm.get_train_loader(ds_name)
            elif ds_name == "corn2":
                loader = dm.get_train_loader(ds_name)
            else:
                loader = dm.get_train_loader(ds_name)

            batch = next(iter(loader))
            print(f"\n{ds_name}: OK")
            for k, v in batch.items():
                if hasattr(v, "shape"):
                    print(f"  {k}: {v.shape} {v.dtype}")
                else:
                    print(f"  {k}: {v}")
        except Exception as e:
            print(f"\n{ds_name}: FAILED — {e}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
