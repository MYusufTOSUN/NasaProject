import numpy as np, pandas as pd
X_tr = np.load("dataset/X_train.npy"); y_tr = np.load("dataset/y_train.npy")
idx  = pd.read_csv("dataset/index_train.csv")
print("X_train:", X_tr.shape, "| pozitif oranı:", (y_tr==1).mean())
print(idx[["snr","depth_ppm"]].describe().round(1))
