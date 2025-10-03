import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from pathlib import Path

def main():
    df = pd.read_csv("index.csv")
    model = YOLO("models/best.pt")
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        img = Path(r["image_path"])
        if not img.exists(): 
            continue
        out = model.predict(source=str(img), verbose=False)
        probs = out[0].probs.data.cpu().tolist()
        pos_prob = probs[1] if len(probs) > 1 else 0.0
        rows.append({
            "target": r["target"],
            "image_path": str(img),
            "label": int(r["label"]),
            "pos_prob": float(pos_prob)
        })
    pd.DataFrame(rows).to_csv("batch_scores.csv", index=False)
    print("[OK] batch_scores.csv written.")

if __name__ == "__main__":
    main()


