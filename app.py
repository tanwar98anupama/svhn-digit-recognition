"""
SVHN Digit Recognition — Flask Web App
Author: [Your Name]
Description: Serves a trained CNN model to predict digits from uploaded images.
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import base64

app = Flask(__name__)
CORS(app)

# ─── Load model once at startup ───────────────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", "svhn_cnn_model.h5")

print(f"Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
model.summary()

# ─── Class labels (0-9) ───────────────────────────────────────────────────────
CLASS_NAMES = [str(i) for i in range(10)]


def preprocess_image(image_bytes):
    """
    Preprocess uploaded image to match training data format:
    - Convert to grayscale
    - Resize to 32x32
    - Normalize pixels to [0, 1]
    - Reshape to (1, 32, 32, 1) for model input
    """
    img = Image.open(io.BytesIO(image_bytes))

    # Convert to grayscale (L mode)
    img = img.convert("L")

    # Resize to 32x32 (model input size)
    img = img.resize((32, 32), Image.LANCZOS)

    # Convert to numpy array and normalize
    img_array = np.array(img, dtype="float32") / 255.0

    # Add batch and channel dimensions: (32, 32) → (1, 32, 32, 1)
    img_array = img_array.reshape(1, 32, 32, 1)

    return img_array


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Accepts: multipart/form-data with 'image' file
    Returns: JSON with predicted digit, confidence, and all class probabilities
    """
    # Validate request
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Include 'image' in form-data."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    # Check file type
    allowed_types = {"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp"}
    if file.content_type not in allowed_types:
        return jsonify({"error": f"Unsupported file type: {file.content_type}. Use JPG, PNG, or WEBP."}), 400

    try:
        # Read image bytes
        image_bytes = file.read()

        # Preprocess
        img_array = preprocess_image(image_bytes)

        # Run model inference
        predictions = model.predict(img_array, verbose=0)
        probabilities = predictions[0]  # Shape: (10,)

        # Get top prediction
        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class])

        # Build response with all class probabilities
        class_probabilities = {
            CLASS_NAMES[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(10)
        }

        # Sort by probability descending for display
        sorted_probs = sorted(
            class_probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return jsonify({
            "predicted_digit": predicted_class,
            "confidence": round(confidence * 100, 2),
            "confidence_label": f"{round(confidence * 100, 1)}%",
            "all_probabilities": class_probabilities,
            "top_3": sorted_probs[:3],
            "status": "success"
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/health")
def health():
    """Health check endpoint for AWS load balancer / monitoring."""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_input_shape": str(model.input_shape)
    })


@app.route("/model-info")
def model_info():
    """Return model architecture details."""
    return jsonify({
        "model_name": "CNN Model 2 — SVHN Digit Recognizer",
        "input_shape": str(model.input_shape),
        "output_classes": 10,
        "dataset": "SVHN (Street View House Numbers)",
        "test_accuracy": "~90%",
        "architecture": "4× Conv2D + BatchNorm + Dropout + Dense"
    })


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    print(f"\n🚀 Starting SVHN Digit Recognition API on port {port}")
    print(f"   Open http://localhost:{port} in your browser\n")
    app.run(host="0.0.0.0", port=port, debug=debug)
