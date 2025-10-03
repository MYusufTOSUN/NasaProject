import argparse, os, shutil
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

OUT_ROOT = Path("data/plots")

def copy_images(df_split, split_name):
    for _, r in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Copy {split_name}"):
        cls = "positive" if r["label"] == 1 else "negative"
        dst_dir = OUT_ROOT / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)
        src = Path(r["image_path"])
        if src.exists():
            dst = dst_dir / f"{r['target']}__{src.name}"
            if not dst.exists():
                shutil.copy2(src, dst)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", type=float, default=0.7)
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    args = p.parse_args()
    assert abs(args.train + args.val + args.test - 1.0) < 1e-6

    df = pd.read_csv("index.csv")  # target,image_path,label,mission,..

    # group by target to keep all its images together
    g = df.groupby("target").agg({"label":"first"}).reset_index()
    y = g["label"].values

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=args.test, random_state=42)
    trainval_idx, test_idx = next(sss1.split(g["target"], y))
    trainval = g.iloc[trainval_idx]
    test = g.iloc[test_idx]

    # val split from trainval
    y_tv = trainval["label"].values
    val_size = args.val / (args.train + args.val)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=42)
    train_idx, val_idx = next(sss2.split(trainval["target"], y_tv))
    train = trainval.iloc[train_idx]
    val = trainval.iloc[val_idx]

    # explode back to per-image rows
    def explode(group_targets):
        return df[df["target"].isin(group_targets["target"])].copy()

    df_train = explode(train)
    df_val = explode(val)
    df_test = explode(test)

    # copy images
    copy_images(df_train, "train")
    copy_images(df_val, "val")
    copy_images(df_test, "test")

    # write data.yaml
    yaml = f"""path: ./data/plots
train: train
val: val
test: test
names:
  0: negative
  1: positive
"""
    (Path("data") / "data.yaml").write_text(yaml, encoding="utf-8")
    print("[OK] Dataset built under data/plots/ and data/data.yaml written.")

if __name__ == "__main__":
    main()


