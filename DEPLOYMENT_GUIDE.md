# 🚀 Deployment Guide — SVHN Digit Recognizer

Complete instructions for GitHub + AWS EC2 deployment.

---

## Part 1 — GitHub (Do this first, takes ~15 min)

### Step 1 — Create the repository

1. Go to https://github.com and log in
2. Click **New repository** (green button, top right)
3. Fill in:
   - **Repository name**: `svhn-digit-recognition`
   - **Description**: `Street digit recognition using CNN — MIT Applied Data Science`
   - **Visibility**: Public ✅
   - Check **Add a README file**
4. Click **Create repository**

---

### Step 2 — Upload your notebook

1. Click **Add file → Upload files**
2. Drag and drop these files:
   - `High_Code_SVHN_Digit_Recognition.ipynb` (your notebook)
   - `README.md` (from the files we created)
   - Any output images: `confusion_matrix.png`, `model_comparison.png`, `sample_images.png`
3. Commit message: `Add completed SVHN digit recognition notebook`
4. Click **Commit changes**

---

### Step 3 — Add the production app files

Upload these files to a subfolder called `app/`:
- `app/app.py`
- `app/requirements.txt`
- `app/templates/index.html`

Or upload them to the root alongside the notebook — either works.

---

### Step 4 — Update README

Edit `README.md` on GitHub (click the pencil icon):
1. Replace `[Your Name]` with your actual name
2. Replace `YOUR_USERNAME` with your GitHub username
3. Add your EC2 URL (after Step 2 below): `Live Demo: http://YOUR-EC2-IP:5000`
4. Commit changes

---

## Part 2 — AWS EC2 Deployment

### Prerequisites
- AWS account (free tier: https://aws.amazon.com/free)
- The saved model file: `svhn_cnn_model.h5`

---

### Step 1 — Launch EC2 Instance

1. Log in to **AWS Console** → search for **EC2** → click **Launch Instance**
2. Settings:
   - **Name**: `svhn-digit-app`
   - **OS**: Ubuntu Server 22.04 LTS (Free tier eligible)
   - **Instance type**: `t2.micro` (free tier)
   - **Key pair**: Click "Create new key pair"
     - Name: `svhn-key`
     - Type: RSA
     - Format: `.pem`
     - Click **Download** — save this file safely (you need it to connect)
   - **Network settings** → click **Edit**:
     - Add inbound rule: **Custom TCP**, Port **5000**, Source **0.0.0.0/0**
     - Keep SSH port 22 already there
3. Click **Launch instance**
4. Wait ~2 minutes for it to show "Running" state

---

### Step 2 — Connect via SSH

```bash
# On your local computer — Mac/Linux Terminal:

# Give the key file correct permissions (REQUIRED)
chmod 400 ~/Downloads/svhn-key.pem

# Connect (replace YOUR_EC2_IP with the Public IPv4 from AWS console)
ssh -i ~/Downloads/svhn-key.pem ubuntu@YOUR_EC2_IP
```

**Windows users**: Use PuTTY or Windows Terminal. Convert .pem to .ppk using PuTTYgen first.

You should see: `ubuntu@ip-xxx-xxx-xxx-xxx:~$`

---

### Step 3 — Install dependencies on EC2

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python pip
sudo apt install python3-pip -y

# Install project dependencies
pip3 install flask flask-cors tensorflow Pillow numpy gunicorn
```

This takes about 5-10 minutes (TensorFlow is large).

---

### Step 4 — Upload files to EC2

Open a **new terminal on your local machine** (keep the SSH session open):

```bash
# Upload app.py
scp -i ~/Downloads/svhn-key.pem app.py ubuntu@YOUR_EC2_IP:~/

# Upload templates folder
scp -i ~/Downloads/svhn-key.pem -r templates/ ubuntu@YOUR_EC2_IP:~/

# Upload your saved model (most important!)
scp -i ~/Downloads/svhn-key.pem svhn_cnn_model.h5 ubuntu@YOUR_EC2_IP:~/
```

---

### Step 5 — Run the app

Back in your SSH session:

```bash
# Check all files are there
ls ~/

# Test run (should see "Running on http://0.0.0.0:5000")
python3 app.py
```

Open your browser: `http://YOUR_EC2_IP:5000`

🎉 **Your app is live!**

---

### Step 6 — Keep it running after you close terminal

```bash
# Stop the current run (Ctrl+C)
# Run with nohup so it keeps running after you disconnect
nohup python3 app.py > app.log 2>&1 &

# Check it's running
ps aux | grep app.py

# See logs
tail -f app.log
```

---

## Testing Your Live App

1. Open `http://YOUR_EC2_IP:5000` in browser
2. Upload any image containing a digit (0-9)
3. Click **Predict Digit**
4. See the predicted digit + confidence + probability bars

**Good test images:**
- Screenshots of house numbers from Google Maps
- Photos of door numbers
- Any digit written on paper

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Connection refused" on port 5000 | Check EC2 Security Group has port 5000 open |
| SSH "Permission denied" | Run `chmod 400 svhn-key.pem` first |
| App crashes with model error | Check `svhn_cnn_model.h5` is in the same folder as `app.py` |
| Slow predictions | t2.micro is CPU-only; normal for TensorFlow, ~1-2 seconds |

---

## What to Post on LinkedIn

> Just deployed my deep learning project live on AWS! 🚀
>
> Built a CNN that recognizes street-level digits (SVHN dataset — same problem Google solves for Maps).
> Results: CNN Model 2 achieved ~90% test accuracy, beating the ANN baseline by 15+ points.
>
> 🔗 Live demo: http://YOUR-EC2-IP:5000 (upload any digit image and see it predicted in real-time)
> 📓 Full notebook + code: github.com/YOUR_USERNAME/svhn-digit-recognition
>
> Stack: Python · TensorFlow · Flask · AWS EC2
> #DeepLearning #CNN #ComputerVision #TensorFlow #AWS #MLOps #MIT

---

*Model trained as part of MIT Applied Data Science Program*
