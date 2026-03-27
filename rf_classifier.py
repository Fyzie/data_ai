import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE # SMOTE creates synthetic examples to balanced the dataset classes

# Load data
database_file_path = "path/to/file.csv"
df = pd.read_csv(database_file_path)

# split features and target
X = df.drop(columns=['target'])
y = df['target']

# train/tst split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# balance the classes using SMOTE 
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# initialize model
clf = RandomForestClassifier(
    n_estimators=500, # increased for better stability
    max_depth=15, 
    class_weight={0: 1, 1: 10}, # tells model that detecting target "1" is 10x time crucial
    random_state=42,
    n_jobs=-1 # use all cpu cores
)

# train
clf.fit(X_res, y_res)

# evaluate binary
# y_pred = clf.predict(X_test)

# Evaluate with threshold
y_probs = clf.predict_proba(X_test)[:, 1]
threshold = 0.20 
y_pred_safe = (y_probs >= threshold).astype(int)

print(f"\nResults with {threshold} Safety Threshold")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_safe))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_safe))

# save model
joblib.dump(clf, "trained_model.pkl")

# visualize feature importance based all decison trees
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nFeature ranking:")
for f in range(X.shape[1]):
    print(f"{f + 1}. {X.columns[indices[f]]}: {importances[indices[f]]:.4f}")
