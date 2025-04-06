import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template_string
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

@app.route('/')
def index():
    return render_template_string("""
    <html>
    <head>
        <title>Live Video</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <script>
            var socket = io.connect('http://' + document.domain + ':' + location.port);
            socket.on('frame', function(data) {
                document.getElementById('video_feed').src = 'data:image/jpeg;base64,' + data;
            });

            socket.on('detected_objects', function(objects) {
                document.getElementById('detected_objects').innerText = "Detected: " + objects;
            });
        </script>
    </head>
    <body>
        <h1>Live Video Feed with Object Detection</h1>
        <p id="detected_objects">Waiting for detections...</p>
        <img id="video_feed" width="640">
    </body>
    </html>
    """)

@socketio.on('video_feed')
def handle_frame(frame):
    try:
        # Decode Base64 frame
        img_data = base64.b64decode(frame)
        np_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Run YOLOv8 object detection
        results = model(img)

        detected_objects = set()
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])  # Class ID
                name = model.names[cls]  # Get class name
                detected_objects.add(name)

                # Draw bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert processed frame back to Base64
        _, buffer = cv2.imencode('.jpg', img)
        frame_base64 = base64.b64encode(buffer).decode()

        # Emit updated frame to the web client
        socketio.emit("frame", frame_base64)

        # Send detected objects back to Raspberry Pi
        detected_objects_str = ", ".join(detected_objects) if detected_objects else "No objects detected"
        socketio.emit("detected_objects", detected_objects_str)

    except Exception as e:
        print(f"❌ Error processing frame: {e}")

if __name__ == "__main__":
    print("🚀 Remote video streaming server started...")
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
