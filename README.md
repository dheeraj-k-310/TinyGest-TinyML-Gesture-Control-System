# ✋ TinyGest  
### TinyML-Based Hand Gesture Recognition for Touchless Application Control

TinyGest is a real-time hand gesture recognition system that enables touchless control of desktop applications using a webcam.  
The system follows TinyML principles by converting hand landmarks into compact feature vectors instead of processing full images.

---

## 🚀 Features

- Real-time webcam-based hand tracking  
- 21 hand landmark detection  
- Lightweight feature-based gesture classification  
- Slide navigation control  
- Media play/pause  
- Volume control  
- Fullscreen toggle  
- TinyML-aligned pipeline  

---

## 🧠 TinyML Approach

Traditional Approach:  
Image → Heavy CNN → Large Model  

TinyGest Approach:  
Image → Hand Landmarks (21 points) → Compact Feature Vector → Lightweight Classifier  

This reduces computational complexity and supports future deployment using:

- TensorFlow Lite  
- TensorFlow Lite Micro (embedded systems)  

---

## 🏗 System Architecture

Webcam Input  
↓  
Hand Landmark Detection (MediaPipe)  
↓  
Feature Extraction  
↓  
Gesture Classification  
↓  
Application Control Trigger  

---

## 📂 Project Structure

ML/
│
├── hand_gesture_control.py
├── hand_landmarker.task
├── list_windows.py
├── test.py
├── requirements.txt
└── README.md


---

## ⚙️ Requirements

- Python 3.10 – 3.12  
- Webcam  
- Windows / Linux  

---

## 📦 Installation

### 1️⃣ Create Virtual Environment

**Windows**
python -m venv myenv
myenv\Scripts\activate


**Linux**
python3 -m venv myenv
source myenv/bin/activate


### 2️⃣ Install Dependencies

pip install opencv-python mediapipe pyautogui numpy


---

## ▶️ Run the Project

python hand_gesture_control.py


Press `q` to exit.

---

## 🧪 Step 5 — Test Gestures

| Gesture      | Expected Result |
|-------------|-----------------|
| ✌ Peace     | Next Slide |
| ☝ Pointing  | Previous Slide |
| 👌 OK       | Play / Pause |
| 🤏 Pinch    | Fullscreen |
| ✋ Open Hand | Volume Up |

Make sure:
- The presentation window is active  
- Webcam clearly detects your hand  
- Lighting conditions are adequate  

---

## 🎓 Future Improvements

- Replace heuristic logic with trained TFLite model  
- Deploy classification on microcontroller using TensorFlow Lite Micro  
- Add GUI dashboard  
- Add custom gesture training  
- Improve gesture threshold tuning  

---

## 📌 License

This project is developed for educational and research purposes.

---

## 👨‍💻 Author

Developed as a TinyML-based gesture recognition prototype for touchless application control.
