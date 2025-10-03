import sys, glob
from pathlib import Path
from ultralytics import YOLO
import pandas as pd

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "models/best.pt"
    test_dir = sys.argv[2] if len(sys.argv) > 2 else "data/plots/test"
    
    if not Path(model_path).exists():
        print(f"Model bulunamadı: {model_path}")
        return
    
    model = YOLO(model_path)
    
    results = []
    for class_name in ["positive", "negative"]:
        folder = Path(test_dir) / class_name
        if not folder.exists():
            print(f"Klasör bulunamadı: {folder}")
            continue
        
        files = list(folder.glob("*.*"))
        print(f"\n{class_name.upper()} örnekleri ({len(files)} adet):")
        
        for f in files[:10]:  # İlk 10 örnek
            r = model.predict(source=str(f), verbose=False)
            pos_prob = float(r[0].probs.data[1]) if len(r[0].probs.data) > 1 else 0.0
            predicted = "positive" if pos_prob > 0.5 else "negative"
            correct = "✓" if predicted == class_name else "✗"
            
            print(f"  {correct} {f.name[:40]:40s} → pos_prob={pos_prob:.4f} ({predicted})")
            results.append({
                "file": f.name,
                "true_class": class_name,
                "pos_prob": pos_prob,
                "predicted": predicted,
                "correct": predicted == class_name
            })
    
    # Özet
    df = pd.DataFrame(results)
    accuracy = (df["correct"].sum() / len(df) * 100) if len(df) > 0 else 0
    print(f"\n{'='*60}")
    print(f"TOPLAM ACCURACY: {accuracy:.2f}% ({df['correct'].sum()}/{len(df)})")
    print(f"{'='*60}")
    
    # Confusion matrix basit
    if len(df) > 0:
        tp = len(df[(df["true_class"] == "positive") & (df["predicted"] == "positive")])
        tn = len(df[(df["true_class"] == "negative") & (df["predicted"] == "negative")])
        fp = len(df[(df["true_class"] == "negative") & (df["predicted"] == "positive")])
        fn = len(df[(df["true_class"] == "positive") & (df["predicted"] == "negative")])
        
        print(f"\nConfusion Matrix:")
        print(f"  True Positive (TP):  {tp}")
        print(f"  True Negative (TN):  {tn}")
        print(f"  False Positive (FP): {fp}")
        print(f"  False Negative (FN): {fn}")

if __name__ == "__main__":
    main()


