# app/app.py
import os
from flask import Flask, request, render_template, redirect, url_for
from joblib import load
from PIL import Image
import numpy as np

app = Flask(__name__, template_folder="templates")
MODEL_PATH = os.path.join("model", "savedmodel.pth")

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")
    payload = load(MODEL_PATH)
    return payload["model"]

model = None

def preprocess_image(file_stream):
    # Open image, convert to grayscale, resize to 64x64, flatten and scale as Olivetti (0..1)
    img = Image.open(file_stream).convert("L")   # grayscale
    img = img.resize((64, 64))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = arr.reshape(1, -1)  # shape (1,4096)
    return arr

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model
    if model is None:
        model = load_model()

    if "image" not in request.files:
        return "No file part 'image' in request", 400

    f = request.files["image"]
    if f.filename == "":
        return "No selected file", 400

    try:
        arr = preprocess_image(f)
        pred = int(model.predict(arr)[0])
        return render_template("upload.html", prediction=pred)
    except Exception as e:
        return f"Error during prediction: {str(e)}", 500

if __name__ == "__main__":
    # For local dev: run with `python app/app.py`
    model = load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)