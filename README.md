# Smart Glasses – Real-Time Object Detection & Face Recognition System

![System Overview](images/SAMPLE.png)

## 📌 Overview

This project implements an **end-to-end real-time vision system** using **Raspberry Pi Zero** and **AWS EC2** for smart glasses.  
The system captures live video from a Raspberry Pi camera, streams it to an AWS server, performs **object detection** and **face recognition**, and sends the results back to the device for **OLED display** and **audio feedback**.

The architecture is modular: **object detection**, **face recognition**, and **image management** run as independent services and are executed one at a time based on the selected use case.

---

## 🧠 System Architecture

**Edge Device (Raspberry Pi Zero):**
- Raspberry Pi Camera Module
- Live video capture and Base64 streaming
- OLED display for detected results
- Audio feedback using `espeak-ng`
- Flask-based image management system

**Cloud (AWS EC2 – Ubuntu):**
- Object detection using **YOLOv8 (custom trained)**
- Face recognition using **MobileFaceNet (Paddle Inference)**
- Flask + Socket.IO for real-time communication

---

## 🔁 Data Flow

1. Raspberry Pi captures frames using `PiCamera`
2. Frames are Base64-encoded and streamed via **Socket.IO**
3. AWS server receives frames and performs:
   - Object detection **OR**
   - Face recognition
4. Processed frames and detected labels are sent back
5. Raspberry Pi:
   - Displays results on OLED
   - Speaks detected labels using text-to-speech

---

## 📂 Project Structure
```
├── obj_stream.py # Streams video from Raspberry Pi to AWS
├── imageManage.py # Local image capture & management (Raspberry Pi)
├── imageServer.py # AWS image upload & embedding creation
├── remote_object_detector.py # AWS YOLOv8 object detection server
├── face_detector.py # AWS face recognition server
├── mobileface_v1.0_infer/ # MobileFaceNet inference model (Paddle)
├── yolov8n.pt # YOLOv8 model (custom trained)
├── haarcascade_frontalface_default.xml
├── templates/ # HTML templates for Flask apps
├── images/ # Stored face images
└── README.md
```




## 🚀 Features

- Real-time video streaming over Socket.IO
- YOLOv8 object detection (custom dataset via LabelImg)
- Face recognition using MobileFaceNet embeddings
- OLED display integration on smart glasses
- Audio feedback for detected objects/faces
- Image upload, delete, and embedding generation
- CPU-only inference (no GPU required)

---

## 🧩 Components Description

### 1️⃣ `obj_stream.py` (Raspberry Pi)
- Captures video using PiCamera
- Streams frames to AWS using Socket.IO
- Receives detected labels
- Displays results on OLED
- Provides audio feedback

---

### 2️⃣ `remote_object_detector.py` (AWS)
- Receives video frames
- Runs YOLOv8 inference
- Draws bounding boxes
- Sends detected object names back to Pi

---

### 3️⃣ `face_detector.py` (AWS)
- Detects faces using Haar Cascade
- Extracts embeddings using MobileFaceNet
- Compares embeddings with stored database
- Recognizes known faces using cosine similarity

---

### 4️⃣ `imageManage.py` (Raspberry Pi)
- Live camera preview
- Capture face images
- Delete images
- Send images to AWS server
- Voice feedback on actions

---

### 5️⃣ `imageServer.py` (AWS)
- Receives uploaded face images
- Extracts face embeddings
- Stores embeddings in `face_database.pkl`
- Manages face dataset lifecycle

---

## 🛠️ Technologies Used

- **Python**
- **Flask**
- **Flask-SocketIO**
- **OpenCV**
- **YOLOv8 (Ultralytics)**
- **PaddlePaddle Inference**
- **MobileFaceNet**
- **Raspberry Pi Camera**
- **OLED (SSD1306)**
- **AWS EC2 (Ubuntu)**

---

## ⚙️ How to Run (High Level)

1. Start **object detection** or **face recognition server** on AWS
2. Run `obj_stream.py` on Raspberry Pi
3. For face enrollment:
   - Use `imageManage.py` on Raspberry Pi
   - Upload images to AWS
   - Generate embeddings via `/extract`
4. Switch between object detection and face recognition by running the corresponding AWS service

---

## ⚠️ Notes

- Only **one detection service** (object or face) runs at a time
- Designed for **CPU-only inference**
- Optimized for low-power edge devices
- Camera is mounted on the **front-right side** of the smart glasses
- Optical system uses:
  - Vertical OLED
  - 45° mirror
  - Lens
  - Transparent projection glass

---

## 📌 Future Improvements

- Run object detection and face recognition in parallel
- Model optimization (INT8 / ONNX)
- Better face detector (RetinaFace / YOLO-face)
- Power optimization for wearable use
- Head-up display UI improvements

---

![](images/final.png)

