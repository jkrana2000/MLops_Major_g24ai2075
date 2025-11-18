# src/test.py
import os
from joblib import load
from sklearn.metrics import accuracy_score

def main():
    model_path = os.path.join("model", "savedmodel.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_path} not found. Run src/train.py first.")
    payload = load(model_path)
    clf = payload["model"]
    X_test = payload["X_test"]
    y_test = payload["y_test"]

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()