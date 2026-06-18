# 🏙️ SVHN Street Digit Recognition

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![AWS EC2](https://img.shields.io/badge/AWS-EC2-yellow.svg)](https://aws.amazon.com/ec2)
[![MIT Program](https://img.shields.io/badge/MIT-Applied%20Data%20Science-red.svg)](https://www.mit.edu)

> Recognizing digits from real-world street photos using deep learning — the same problem Google solves to power Maps address recognition. Trained on the SVHN dataset, deployed as a live web app on AWS EC2.

🔗 Live Demo: Coming soon — deploying on AWS EC2 ← upload any digit image

---

## 📊 Results

| Model | Architecture | Test Accuracy |
|---|---|---|
| ANN | Dense(64) → Dense(32) → Softmax | ~77% |
| CNN Model 1 | 2× Conv2D + MaxPool + Dense | ~85% |
| **CNN Model 2 ⭐** | **4× Conv2D + BatchNorm + Dropout** | **~90%** |

CNN Model 2 outperforms the ANN baseline by **13 percentage points** by learning spatial features (edges, curves, stroke patterns) that a flat neural network cannot detect.

---

## Model Architecture
The final model (`CNN Model 2`) utilizes a sequential convolutional neural network design optimized to handle the variance in lighting, blur, and distortion present in real-world street digit images.

### Layer Breakdown
* **Input Layer:** $32 \times 32 \times 3$ RGB images.
* **Convolutional Block 1:**
  * 2× `Conv2D` layers (32 filters, $3 \times 3$ kernel) to capture low-level features like edges and orientations.
  * `LeakyReLU(alpha=0.1)` activation to prevent dying neurons and retain negative gradient information.
  * `BatchNormalization` to stabilize activations and accelerate training convergence.
  * `MaxPooling2D` ($2 \times 2$) to reduce spatial dimensions and introduce translation invariance.
  * `Dropout(0.25)` to prevent early overfitting.
* **Convolutional Block 2:**
  * 2× `Conv2D` layers (64 filters, $3 \times 3$ kernel) to learn high-level spatial combinations (curves, shapes, digit contours).
  * `LeakyReLU(alpha=0.1)` activation for robust non-linear feature mapping.
  * `BatchNormalization` to ensure stable gradient flow through deeper layers.
  * `MaxPooling2D` ($2 \times 2$) to compress representation.
  * `Dropout(0.3)` to regularize feature extraction.
* **Classification Head:**
  * `Flatten` layer to convert 2D feature maps into a 1D feature vector.
  * `Dense` layer (128 units) with `LeakyReLU(alpha=0.1)` for multi-feature interaction.
  * `Dropout(0.5)` for aggressive regularization before classification.
  * `Dense` output layer (10 units, `Softmax` activation) yielding a probability distribution across digits $0\text{–}9$.

### Optimization & Training Parameters
* **Loss Function:** Categorical Cross-Entropy
* **Optimizer:** Adam Optimizer
* **Regularization:** Combined Batch Normalization, LeakyReLU, and layered Dropout ($25\%\text{–}50\%$)

---

## 🧠 Key Concepts

- **Why CNNs beat ANNs on images** — flattening destroys spatial relationships
- **LeakyReLU** — prevents dying neurons (negative gradients allowed)
- **BatchNormalization** — stabilizes training, faster convergence
- **Dropout(0.5)** — prevents overfitting by randomly disabling neurons
- **Confusion analysis** — digits 3↔5 and 1↔7 are hardest to separate

---

## 🏗️ Project Structure

```
svhn-digit-recognition/
├── SVHN_Street_Digit.ipynb
├── README.md
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```

---

## 🖥️ Running Locally

```bash
git clone https://github.com/tanwar98anupama/svhn-digit-recognition
cd svhn-digit-recognition

pip install -r requirements.txt

cp /path/to/svhn_cnn_model.h5 ./

python app.py
# Open http://localhost:5000
```

---

## ☁️ Production Deployment (AWS EC2)

See [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for the complete step-by-step AWS setup.

**Architecture:**
```
User browser
     ↓  HTTP
AWS EC2 (Ubuntu t2.micro)
     ↓  
Flask app (port 5000)
     ↓  
TensorFlow CNN model
     ↓  
Prediction JSON response
```

---

## 📁 API Reference

**`POST /predict`** — Predict a digit from an image

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@your_image.jpg"
```

Response:
```json
{
  "predicted_digit": 7,
  "confidence": 94.3,
  "confidence_label": "94.3%",
  "all_probabilities": {"0": 0.1, "1": 1.2, ..., "7": 94.3, ...},
  "status": "success"
}
```

**`GET /health`** — Health check  
**`GET /model-info`** — Model architecture details

---

## 🛠️ Tech Stack

Python · TensorFlow/Keras · Flask · NumPy · Pillow · AWS EC2 · Ubuntu 22.04

---

## 👤 Author

**[Anupama Rathod]**   
[LinkedIn](https://www.linkedin.com/in/anupama-rathod/) · [GitHub](https://github.com/tanwar98anupama)
