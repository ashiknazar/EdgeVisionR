import time
import base64
import socketio
import cv2
import numpy as np
from picamera import PiCamera
from io import BytesIO
import signal
import sys
import threading
import os

# OLED Display Libraries
try:
    import board
    import busio
    from adafruit_ssd1306 import SSD1306_I2C
    from PIL import Image, ImageDraw, ImageFont
    OLED_AVAILABLE = True
except ImportError:
    print("⚠️ OLED library not found. Detected objects will be printed in the terminal.")
    OLED_AVAILABLE = False

# Global OLED variables
disp = None
font = None


def initialize_oled():
    """ Initializes the OLED display if available. """
    global disp, font, OLED_AVAILABLE
    if OLED_AVAILABLE:
        try:
            print("⏳ Initializing OLED...")
            i2c = busio.I2C(board.SCL, board.SDA)
            disp = SSD1306_I2C(128, 64, i2c)  # Ensure this matches your OLED size (128x32 or 128x64)
            disp.fill(0)
            disp.show()
            font =ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
            print("✅ OLED Initialized Successfully!")
        except Exception as e:
            print(f"❌ OLED initialization failed: {e}")
            OLED_AVAILABLE = False

# Run OLED initialization in a separate thread
oled_thread = threading.Thread(target=initialize_oled, daemon=True)
oled_thread.start()

# Initialize Flask-SocketIO client
sio = socketio.Client()
REMOTE_SERVER_URL = "http://18.209.57.101:5000"

@sio.on("detected_objects")
def receive_detected_objects(data):
    """ Displays detected objects on OLED or prints to terminal. """
    if OLED_AVAILABLE and disp is not None:
        try:
            # Create image canvas
            image = Image.new("1", (disp.width, disp.height))
            draw = ImageDraw.Draw(image)
            draw.text((0, 0), f"{data}", font=font, fill=255)
            disp.image(image)
            disp.show()  # Use .show() instead of .display()
            os.system("espeak-ng '{data}'")
        except Exception as e:
            print(f"⚠️ OLED display error: {e}")
    else:
        print(f"📢 Detected Objects: {data}")

def stop_signal_handler(sig, frame):
    """ Handles SIGTERM & SIGINT to stop streaming gracefully. """
    global camera, sio
    print("🛑 Received stop signal, stopping streaming...")

    try:
        if 'camera' in globals() and camera is not None:
            camera.close()
            print("📷 Camera closed.")

        if sio.connected:
            sio.disconnect()
            print("🔌 SocketIO disconnected.")

    except Exception as e:
        print(f"⚠️ Error during cleanup: {e}")

    sys.exit(0)  # Exit gracefully

# Register the SIGTERM and SIGINT (Ctrl+C) handlers
signal.signal(signal.SIGTERM, stop_signal_handler)
signal.signal(signal.SIGINT, stop_signal_handler)

def send_frames():
    """ Captures and streams frames from PiCamera to the remote server. """
    camera = PiCamera()
    camera.resolution = (320, 240)
    camera.framerate = 10
    stream = BytesIO()

    try:
        sio.connect(REMOTE_SERVER_URL)
        print("✅ Connected to remote server!")

        while True:
            camera.capture(stream, format="jpeg", use_video_port=True)
            img_base64 = base64.b64encode(stream.getvalue()).decode("utf-8")
            sio.emit("video_feed", img_base64)

            stream.seek(0)
            stream.truncate()
            time.sleep(0.1)

    except Exception as e:
        print(f"❌ ERROR: {e}")

    finally:
        camera.close()
        sio.disconnect()

@sio.on("stop_stream")
def stop_stream():
    """ This function is triggered when Flask sends a stop signal. """
    global camera, sio
    try:
        import subprocess
        subprocess.run(["pkill", "-f", "local_stream.py"])  # Kill the running process
        return "🛑 Streaming stopped!", 200
    except Exception as e:
        return f"❌ Error: {str(e)}", 500
    if camera:
        camera.close()

if __name__ == "__main__":
    send_frames()