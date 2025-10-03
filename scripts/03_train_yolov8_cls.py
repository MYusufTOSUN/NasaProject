import argparse, os, sys
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/plots", help="Ultralytics data path (dir with train/val/test)")
    ap.add_argument("--model", default="yolov8n-cls.pt", help="Base cls model (yolov8n/s/m/l/x-cls.pt)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--project", default="runs", help="Ultralytics project dir")
    ap.add_argument("--name", default="exp_exo", help="Run name")
    ap.add_argument("--device", default="", help="''=auto, '0'=cuda:0, 'cpu'=force CPU")
    args = ap.parse_args()

    if not Path(args.data).exists():
        print(f"[!] data.yaml not found: {args.data}")
        sys.exit(1)

    print(f"[i] Training {args.model} on {args.data} for {args.epochs} epochs...")
    cmd_kwargs = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                      project=args.project, name=args.name, device=args.device)
    # Start training
    model = YOLO(args.model)
    model.train(**cmd_kwargs)

    # Find best.pt and copy to models/best.pt
    # Ultralytics places weights under runs/classify/<name*>/weights/best.pt
    runs = sorted(Path(args.project, "classify").glob(args.name + "*"), key=os.path.getmtime, reverse=True)
    if not runs:
        print("[!] Could not locate run directory under runs/classify/")
        sys.exit(1)
    best = runs[0] / "weights" / "best.pt"
    if not best.exists():
        print(f"[!] best.pt not found at {best}")
        sys.exit(1)

    Path("models").mkdir(parents=True, exist_ok=True)
    out = Path("models/best.pt")
    out.write_bytes(best.read_bytes())
    print(f"[OK] Copied best weights to {out.resolve()}")

    # Optional: quick test on test set (if exists)
    test_dir = Path("data/plots/test/positive")
    if test_dir.exists() and any(test_dir.iterdir()):
        print("[i] Quick sanity-check on a few test images...")
        model = YOLO(str(out))
        model.predict(source=str(test_dir), save=False, max_det=1, imgsz=args.imgsz, device=args.device, verbose=False)
        print("[OK] Prediction run finished.")


if __name__ == "__main__":
    main()


