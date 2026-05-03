"""Fine-tune YOLOv8n on a drone-detection dataset.

Defaults are tuned for hackathon iteration on Apple-Silicon MPS:
  - imgsz=640 (matches the Seraphim dataset's native pad)
  - batch=32 (yolov8n on M-series MPS handles this comfortably; reduce if OOM)
  - epochs=30 (diminishing returns past this for a single-class detector)
  - device=mps when available, else cpu (CUDA is auto-picked too if present)
  - project=runs/train, name=drone_v{N}  (auto-incremented per run)

The trained weights land at:  runs/train/<name>/weights/best.pt
A `latest_drone_best.pt` symlink in the project root always points at the
most recent best.pt — copy that to the Jetson when ready.

Usage:
    python scripts/train_drone_yolo.py --data datasets/seraphim_drone/data.yaml
    python scripts/train_drone_yolo.py --data ... --epochs 50 --batch 16
    python scripts/train_drone_yolo.py --data ... --fraction 0.25  # use 25% of train images
    python scripts/train_drone_yolo.py --data ... --resume         # resume last interrupted run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="Path to YOLO data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="Base weights (downloaded if missing)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of training images to use (1.0 = all). Useful for fast iteration.",
    )
    p.add_argument(
        "--device",
        default="auto",
        help="'auto' | 'mps' | 'cpu' | '0' (CUDA index).  auto picks mps if available, else cpu.",
    )
    p.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default=None, help="Run name (default: auto-increment drone_v1, drone_v2, ...)")
    p.add_argument("--patience", type=int, default=15, help="Early-stop patience (epochs without val improvement)")
    p.add_argument("--resume", action="store_true", help="Resume the last interrupted run")
    p.add_argument("--cache", default="ram", choices=["ram", "disk", "false"], help="Cache images for speed")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _autoname(project: str) -> str:
    proj = Path(project)
    proj.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in proj.iterdir() if p.is_dir()}
    i = 1
    while f"drone_v{i}" in existing:
        i += 1
    return f"drone_v{i}"


def _link_latest(best_pt: Path) -> None:
    if not best_pt.exists():
        return
    link = Path("latest_drone_best.pt")
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(best_pt.resolve())
        print(f"[link] latest_drone_best.pt -> {best_pt}")
    except OSError as e:
        print(f"[link] could not symlink ({e}); copying instead")
        link.write_bytes(best_pt.read_bytes())


def main() -> int:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[error] data.yaml not found: {data_path}", file=sys.stderr)
        print("  Run: python scripts/download_drone_dataset.py", file=sys.stderr)
        return 1

    device = _pick_device(args.device)
    name = args.name or _autoname(args.project)
    cache = False if args.cache == "false" else args.cache

    print("=" * 60)
    print(f"  device       {device}")
    print(f"  base model   {args.model}")
    print(f"  data         {data_path}")
    print(f"  epochs       {args.epochs}")
    print(f"  imgsz        {args.imgsz}")
    print(f"  batch        {args.batch}")
    print(f"  fraction     {args.fraction}")
    print(f"  workers      {args.workers}")
    print(f"  cache        {cache}")
    print(f"  project/name {args.project}/{name}")
    print("=" * 60, flush=True)

    # Lazy-import so --help works without ultralytics/torch installed.
    from ultralytics import YOLO  # type: ignore

    # MPS uses up unified memory aggressively; keep CPU fallback enabled for
    # any ops not implemented on MPS so we don't crash mid-epoch.
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=args.project,
        name=name,
        fraction=args.fraction,
        patience=args.patience,
        resume=args.resume,
        cache=cache,
        seed=args.seed,
        verbose=True,
        plots=True,
    )

    save_dir = Path(getattr(results, "save_dir", Path(args.project) / name))
    best_pt = save_dir / "weights" / "best.pt"
    print(f"\n[done] best weights: {best_pt}")
    _link_latest(best_pt)
    print("\nNext:")
    print(f"  Verify locally:   python scripts/verify_yolo.py --weights {best_pt}")
    print(f"  Push to Jetson:   scp {best_pt} cask:~/Katena/drone_v1.pt")
    print( "  Run on Jetson:    python3 scripts/jetson_live_detect.py --weights drone_v1.pt --conf 0.4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
