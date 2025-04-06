import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import pickle
import paddle
import paddle.inference as paddle_infer

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load PaddlePaddle MobileFaceNet Model
MODEL_PATH = "mobileface_v1.0_infer/inference.pdmodel"

PARAMS_PATH = "mobileface_v1.0_infer/inference.pdiparams"
FACE_CASCADE_PATH = "haarcascade_frontalface_default.xml"
FACE_DATABASE_PATH = "face_database.pkl"

# Load Haarcascade for Face Detection
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# Load Face Embeddings Database
with open(FACE_DATABASE_PATH, "rb") as f:
    face_database = pickle.load(f)  # Dictionary { "name": numpy_embedding }

# Initialize Paddle Inference
config = paddle_infer.Config(MODEL_PATH, PARAMS_PATH)
config.disable_gpu()  # Ensure CPU usage
predictor = paddle_infer.create_predictor(config)
input_handle = predictor.get_input_handle(predictor.get_input_names()[0])

output_handle = predictor.get_output_handle(predictor.get_output_names()[0])

def preprocess_face(face_img):
    
    face_img = cv2.resize(face_img, (112, 112))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0  # Normalize
    face_img = np.transpose(face_img, (2, 0, 1))  # Convert to CHW format
    return np.expand_dims(face_img, axis=0)  # Add batch dimension


def extract_embedding(face_img):
    processed_face = preprocess_face(face_img)
    input_handle.copy_from_cpu(processed_face)
    predictor.run()
    embedding = output_handle.copy_to_cpu().squeeze()
    
    return embedding / np.linalg.norm(embedding)   # Normalize embedding


def recognize_face(face_img):
    """Compare extracted embedding with database and return recognized name."""
    embedding = extract_embedding(face_img)

    best_match = "Unknown"
    max_similarity = -1 

    for name, saved_embedding in face_database.items():
        similarity = np.dot(embedding, saved_embedding) # Cosine Similarity
        if similarity > max_similarity:
            max_similarity = similarity
            best_match = name

    return best_match if max_similarity > 0.5 else "Unknown"  # Set threshold for recognition

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Live Face Recognition</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            var socket = io.connect('http://' + document.domain + ':' + location.port);
            socket.on('frame', function(data) {
                document.getElementById('video_feed').src = 'data:image/jpeg;base64,' + data;
            });

            socket.on('detected_objects', function(data) {
                document.getElementById('detected_faces').innerText = "Recognized: " + data;
            });
        </script> </head>
    <body>
        <h1>Live Face Recognition</h1>
        <p id="detected_faces">Waiting for faces...</p>
        <img id="video_feed" width="640">
    </body>
    </html>
    """)
@socketio.on('video_feed')
def handle_frame(frame):
    try:
        # Decode Base64 Frame
        img_data = base64.b64decode(frame)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Convert to Grayscale for Face Detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # Detect Faces using Haarcascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(20, 20))

        recognized_faces = []

        for (x, y, w, h) in faces:
            padding = 20  # Increase the face region by 10 pixels
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
            face = img[y:y+h, x:x+w]  # Extract Face ROI


            name = recognize_face(face)  # Recognize Face

            recognized_faces.append(name)

            # Draw Bounding Box and Name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert Processed Frame Back to Base64
        _, buffer = cv2.imencode('.jpg', img)
        frame_base64 = base64.b64encode(buffer).decode()
        # Emit Updated Frame to Web Client
        socketio.emit("frame", frame_base64)

        # Send Recognized Names to Local Flask App
        recognized_faces_str = ", ".join(recognized_faces) if recognized_faces else "No faces detected"
        socketio.emit("detected_objects", recognized_faces_str,include_self=False)

    except Exception as e:
        print(f"❌ Error processing frame: {e}")
if __name__ == "__main__":
    print("🚀 Remote face recognition server started...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
