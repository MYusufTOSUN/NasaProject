import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Data load
X_train = np.load("dataset/X_train.npy")
y_train = np.load("dataset/y_train.npy")
X_val   = np.load("dataset/X_val.npy")
y_val   = np.load("dataset/y_val.npy")
X_test  = np.load("dataset/X_test.npy")
y_test  = np.load("dataset/y_test.npy")

print("Shapes:", X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=2000)
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_val)
y_proba = logreg.predict_proba(X_val)[:,1]

print("LogReg Validation Report:")
print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_val)
y_proba = rf.predict_proba(X_val)[:,1]

print("RandomForest Validation Report:")
print(classification_report(y_val, y_pred))
print("ROC-AUC:", roc_auc_score(y_val, y_proba))
