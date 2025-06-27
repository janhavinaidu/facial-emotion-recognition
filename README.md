
# 🎭 Facial Emotion Recognition using Deep Learning

This project detects **real-time facial emotions** using a **PyTorch CNN model** integrated with OpenCV and deployed using a webcam. It supports 5 emotion classes:

> `Angry 😠`, `Smiling 😄`, `Laughing 😂`, `Neutral 😐`, `Crying 😢`

---

## 📸 Demo Preview

> 🎥 Real-time emotion prediction using webcam  
> ✅ Works with subtle and exaggerated facial expressions  
> ⚡ Fast inference with TorchScript-optimized model

---

## 🚀 Features

- 🔍 Real-time face detection using OpenCV
- 🧠 Emotion classification with a deep CNN model
- 🏷️ 5-Class support: Angry, Happy, Laugh, Sad, Neutral
- 🎯 Trained with class-weighted loss and heavy augmentation
- 📦 Lightweight TorchScript model for mobile integration

---

## 📁 Project Structure


Export Format	TorchScript (.pt)


Inference	Real-time via webcam

facial-emotion-recognition/
│
├── app.py # Real-time webcam emotion detection
├── train.py # Training script for the CNN model
├── test.py # TorchScript model testing script
├── models.pt # TorchScript-optimized trained model ✅
├── label.txt # Emotion labels used by the model
├── requirements.txt # All Python dependencies
├── .gitignore # Ignored folders like datasets, cache
└── README.md # This file 😊


---

## 🧠 Model Details

- CNN with 4 Conv layers + BatchNorm + Dropout
- Fully connected layers with regularization
- Heavy data augmentation (rotation, flip, affine, color jitter)
- Class weights handled for imbalanced dataset
- Exported via `torch.jit.trace()` as `models.pt` ✅

---

## 🛠️ Setup & Run

### 1. Clone the repository

```bash
git clone https://github.com/janhavinaidu/facial-emotion-recognition.git
cd facial-emotion-recognition

python -m venv venv
venv\Scripts\activate     # On Windows
# source venv/bin/activate   # On macOS/Linux

pip install -r requirements.txt

python test.py


