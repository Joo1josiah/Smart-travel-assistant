from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import os
from werkzeug.utils import secure_filename
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Load pre-trained CNN model
MODEL_PATH = "models/cnn_model.h5"
model = load_model(MODEL_PATH)

# Load recommendation dataset
RECOMMENDATION_DATA_PATH = "models/recommendation_data.pkl"
with open(RECOMMENDATION_DATA_PATH, "rb") as f:
    recommendation_data = pickle.load(f)

# Allowed image extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    # Process the image with the ML model
    category = "beach"  # Replace with actual ML classification logic
    recommendations = ["Place A", "Place B", "Place C"]  # Dummy recommendations

    return jsonify({"category": category, "recommendations": recommendations})
@app.route('/')
def home():
    return "Welcome to Smart Travel Assistant API! Available endpoints: /upload (POST)"


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

