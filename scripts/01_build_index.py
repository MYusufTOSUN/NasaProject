import os, re, sys, argparse
import pandas as pd
from pathlib import Path

RAW_DIR_DEFAULT = "scripts/raw_images"  # change if needed

# Resolve repository root (this file is under <repo>/scripts/)
REPO_ROOT = Path(__file__).resolve().parents[1]

def find_images(raw_dir: Path):
    exts = {".png", ".jpg", ".jpeg"}
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if Path(f).suffix.lower() in exts:
                yield Path(root) / f

def norm(s: str) -> str:
    return re.sub(r"\s+", "", s.lower())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default=os.environ.get("RAW_DIR", ""), help="Raw images directory (overrides default)")
    args = parser.parse_args()

    # Determine raw images directory robustly
    raw_input = args.raw.strip() if isinstance(args.raw, str) else ""
    if raw_input:
        raw_dir = Path(raw_input)
    else:
        raw_dir = REPO_ROOT / RAW_DIR_DEFAULT

    if not raw_dir.is_absolute():
        raw_dir = (REPO_ROOT / raw_dir).resolve()

    if not raw_dir.exists():
        print(f"[!] RAW images folder not found: {raw_dir}")
        print("    Create it or set with --raw <dir> or RAW_DIR env var, or change RAW_DIR_DEFAULT.")
        sys.exit(1)

    targets_path = REPO_ROOT / "targets.csv"
    if not targets_path.exists():
        print(f"[!] targets.csv not found at: {targets_path}")
        sys.exit(1)

    targets = pd.read_csv(targets_path)  # columns: target,mission,label
    targets["key"] = targets["target"].astype(str).str.lower().str.replace(r"\s+", "", regex=True)

    rows = []
    for img in find_images(raw_dir):
        name = img.name.lower()
        # match any target that is substring of file name
        matches = targets[targets["key"].apply(lambda k: k in name)]
        if matches.empty:
            # no target found; skip silently
            continue
        for _, r in matches.iterrows():
            is_binned = int("_binned" in name or "binned" in name)
            is_phase = int("_phase" in name or "phase" in name)
            rows.append({
                "target": r["target"],
                "mission": r["mission"],
                "label": int(r["label"]),
                "image_path": str(img.as_posix()),
                "is_binned": is_binned,
                "is_phase": is_phase
            })

    if not rows:
        print("[!] No matches between targets.csv and image filenames.")
        sys.exit(1)

    df = pd.DataFrame(rows).drop_duplicates(subset=["target","image_path"])
    out_index = REPO_ROOT / "index.csv"
    df.to_csv(out_index, index=False)
    print(f"[OK] index.csv written with {len(df)} rows at {out_index}")

if __name__ == "__main__":
    main()


