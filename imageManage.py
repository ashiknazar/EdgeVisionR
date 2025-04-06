from flask import Flask, render_template, request, redirect, url_for, Response
from picamera import PiCamera
import time
import os
import threading
import io
import requests

import subprocess

import atexit
import signal


app = Flask(__name__)

# Ensure 'static' folder exists
if not os.path.exists("static"):
    os.makedirs("static")

EC2_URL="http://44.223.28.223:5000/upload"

camera = None
streaming = False  # Global flag to manage streaming state
stream_lock = threading.Lock()  # Ensures thread-safe operations
def init_camera():
    """Safely initialize the camera when needed."""
    global camera
    with stream_lock:
        if camera is None:
            camera = PiCamera()
            camera.resolution = (320, 240)
            camera.framerate = 10
            time.sleep(2)  # Let the camera adjust

def release_camera():
    """Properly release the camera resource."""
    global camera
    with stream_lock:
        if camera:
            try:
                camera.close()
                print("Camera released successfully")
            except Exception as e:
                print(f"Error closing camera: {e}")
            finally:
                camera = None
                time.sleep(1)
# Import for in-memory streaming


def generate_frames():
    """Continuously capture frames from PiCamera and send as MJPEG stream."""
    global streaming
    init_camera()

    stream = io.BytesIO()  # Use an in-memory stream

    while streaming:
        with stream_lock:
            camera.capture(stream, format="jpeg", use_video_port=True)
            stream.seek(0)
            frame_bytes = stream.read()
            stream.seek(0)
            stream.truncate()  # Clear the stream for the next frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.1)  # Adjust frame rate if needed

@app.route('/')
def index():
    return render_template('imIndex.html')

@app.route('/manage_images')
def manage_images():
    """List saved images in the static folder."""
    images = [f for f in os.listdir("static") if f.endswith(".jpg")]
    return render_template('manage_img.html', images=images)

@app.route('/start_stream', methods=['POST'])
def start_stream():
    """Start live streaming."""
    global streaming
    streaming = True
    return "Streaming Started", 200

@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    """Stop live streaming and release camera."""
    global streaming
    streaming = False
    release_camera()
    return "Streaming Stopped", 200

@app.route('/video_feed')
def video_feed():
    """Route to serve the video feed."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/capture', methods=['POST'])
def capture():
    """Capture an image safely."""
    global streaming
    streaming = False  # Ensure streaming stops
    release_camera()  # Release camera first

    init_camera()  # Reinitialize after releasing

    image_name = request.form.get('image_name', '').strip()
    if not image_name:
        return "Error: Please enter a valid image name.", 400

    image_name = image_name.replace(" ", "_") + ".jpg"
    image_path = f'static/{image_name}'

    with stream_lock:
        camera.capture(image_path)

    release_camera()  # Release after capturing

    return redirect(url_for('manage_images'))

@app.route('/delete_image/<image_name>', methods=['POST'])
def delete_image(image_name):
    """Delete an image from the static folder."""
    image_path = os.path.join("static", image_name)

    if os.path.exists(image_path):
        os.remove(image_path)
        return {"success": True}
    else:
        return {"success": False, "error": "File not found"}, 404

@app.route('/send_images')
def send_images():
    """Send images from static folder to EC2 server."""
    image_folder = "static"
    files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

    for image in files:
        file_path = os.path.join(image_folder, image)
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(EC2_URL, files=files)
            print(response.json())
    os.system("espeak-ng 'image sent'")
    return "Images sent to EC2", 200
def on_exit():
    os.system("espeak-ng 'goodbye'")

# Run the exit function when the script stops
atexit.register(on_exit)

# Handle SIGINT (Ctrl+C) and SIGTERM (kill command)
def handle_signal(signum, frame):
    print("\n🛑 Flask app is shutting down...")
    on_exit()  # Call exit function
    exit(0)

signal.signal(signal.SIGINT, handle_signal)  # Handle Ctrl+C
signal.signal(signal.SIGTERM, handle_signal)

if __name__ == '__main__':
    os.system("espeak-ng 'welcome to image management'")
    app.run(host='0.0.0.0', port=5000, debug=True)
