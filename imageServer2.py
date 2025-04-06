from flask import Flask, request, jsonify, render_template,url_for
import os
import  cv2
import numpy as np
import pickle
import paddle.inference as paddle_infer


app = Flask(__name__)

UPLOAD_FOLDER = "static/images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
HAARCASCADE_PATH ="haarcascade_frontalface_default.xml"
face_detector = cv2.CascadeClassifier(HAARCASCADE_PATH)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (112, 112))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img.astype("float32") / 255.0 
    face_img = np.transpose(face_img, (2, 0, 1))  
    return np.expand_dims(face_img, axis=0)  

def get_embedding(face_input, predictor):
    input_names = predictor.get_input_names()
    input_tensor = predictor.get_input_handle(input_names[0])
    input_tensor.copy_from_cpu(face_input)
    predictor.run()
    output_names = predictor.get_output_names()
    output_tensor = predictor.get_output_handle(output_names[0])
    embedding = output_tensor.copy_to_cpu()
    return embedding


@app.route('/')
def index():
    """Displays the upload page."""
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Receives and saves uploaded images."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200



def create_face_database(image_folder="images", pickle_file="face_database.pkl", predictor=None):
    face_db = {}
    image_folder=os.path.join("static",image_folder)
    
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read {image_name}. Skipping.")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        if len(faces) == 0:
            print(f"No face detected in {image_name}. Skipping.")
            continue
        # Assume the first detected face is the target face
        (x, y, w, h) = faces[0]
        padding = 10
        x, y = max(0, x - padding), max(0, y - padding)
        w, h = min(img.shape[1] - x, w + 2 * padding), min(img.shape[0] - y, h + 2 * padding)
        face_roi = img[y:y+h, x:x+w]
        face_preprocessed = preprocess_face(face_roi)
        embedding = get_embedding(face_preprocessed, predictor)

        # Use the filename (without extension) as the unique identifier
        name = os.path.splitext(image_name)[0]
        embedding = embedding.flatten() 
        face_db[name] = embedding / np.linalg.norm(embedding) # Convert to (128,)
  # Normalize to unit vector

        print(f"Processed {image_name} and added to database.")
    with open(pickle_file, "wb") as f:
        pickle.dump(face_db, f)
    print(f"Face database saved to {pickle_file}")

def load_predictor(model_dir="mobileface_v1.0_infer"):
    model_file = os.path.join(model_dir, "inference.pdmodel")
    params_file = os.path.join(model_dir, "inference.pdiparams")
    config = paddle_infer.Config(model_file, params_file)
    config.disable_gpu()  # Use CPU; change if you have GPU available
    predictor = paddle_infer.create_predictor(config)
    return predictor


@app.route('/extract')
def extract():
    predictor = load_predictor("mobileface_v1.0_infer")
    create_face_database(image_folder="images", pickle_file="face_database.pkl", predictor=predictor)
    return "Embeddings created for all images!"



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
