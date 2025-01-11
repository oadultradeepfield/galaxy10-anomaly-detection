import numpy as np
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from helper import (compute_reconstruction_errors, create_feature_extractor,
                    load_images, load_model, resize_image, validate_apikey)
from models import Autoencoder

load_dotenv()

app = Flask(__name__)

threshold = 0.03526
feature_extractor = create_feature_extractor()
model = Autoencoder()
model = load_model(model)

@app.route('/detect-anomalies', methods=['POST'])
def detect_anomalies():
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return jsonify({"error": "Authorization header missing or malformed"}), 401

    apikey = auth_header.split("Bearer ")[1]
    is_valid, message = validate_apikey(apikey)

    if not is_valid:
        return jsonify({"error": message}), 401

    image_paths = request.json.get("image_paths", [])
    if not image_paths:
        return jsonify({"error": "No image paths provided"}), 400

    img_list = load_images(image_paths)
    processed_img_list = [resize_image(img) for img in img_list]

    tensor_images = torch.stack([
        torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255
        for img in processed_img_list
    ])
    
    features = feature_extractor(tensor_images).squeeze()
    reconstruction_errors = compute_reconstruction_errors(model, features)

    results = []
    for path, error in zip(image_paths, reconstruction_errors):
        results.append({
            "filename": path,
            "loss": error.item(),
            "anomaly": error.item() > threshold
        })

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=False)
