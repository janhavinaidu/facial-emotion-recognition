import cv2
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

# ===== Load TorchScript model =====
model = torch.jit.load("models.pt")
model.eval()

# ===== Load labels =====
with open("labels.txt", "r", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]

# ===== Face detection using Haar Cascade =====
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ===== Preprocessing =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ===== Start Webcam =====
cap = cv2.VideoCapture(0)

print("🎥 Starting webcam. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_pil = cv2.resize(face, (48, 48))  # Resize to match model input
        face_tensor = transform(Image.fromarray(face_pil)).unsqueeze(0)

        # Predict emotion
        with torch.no_grad():
            output = model(face_tensor)
            probs = torch.softmax(output[0], dim=0)
            pred_idx = torch.argmax(probs).item()
            label = labels[pred_idx]

        # Draw on frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Show frame
    cv2.imshow("Facial Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
