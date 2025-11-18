# src/train.py
import os
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from joblib import dump
import numpy as np

def main():
    os.makedirs("model", exist_ok=True)
    data = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = data.data               # shape (400, 4096)  (64x64 flattened)
    y = data.target             # shape (400,)
    # 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.7, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    model_path = os.path.join("model", "savedmodel.pth")
    dump({
        "model": clf,
        "X_test": X_test,
        "y_test": y_test
    }, model_path)
    print(f"Saved model + test-split to {model_path}")

if __name__ == "__main__":
    main()