from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import joblib
import numpy as np
import os

# ----------------------- Initialize Flask -----------------------
app = Flask(__name__)

# ------------------- Model Configuration -------------------
MODEL_DIR = "models"
TORCH_MODELS = ['ann', 'cnn']
SKLEARN_MODELS = ['knn', 'svm', 'decisiontree', 'randomforest']
ALL_MODELS = TORCH_MODELS + SKLEARN_MODELS

# ------------------- Image Preprocessing -------------------
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

# ------------------- Model Definitions -------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(64 * 64, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 64 * 64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> [B, 8, 32, 32]
        x = self.pool(F.relu(self.conv2(x)))  # -> [B, 16, 16, 16]
        x = x.view(-1, 16 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ------------------- Load Model -------------------
def load_model(model_name):
    model_name = model_name.lower()
    if model_name in TORCH_MODELS:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
        model = ANN() if model_name == 'ann' else CNN()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model, 'torch'
    elif model_name in SKLEARN_MODELS:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")
        model = joblib.load(model_path)
        return model, 'sklearn'
    return None, None

# ------------------- Predict Route -------------------
@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form.get('model')
    image_file = request.files.get('image')

    if not model_name or model_name.lower() not in ALL_MODELS:
        return jsonify({'error': 'Invalid or missing model name'}), 400
    if not image_file:
        return jsonify({'error': 'Image file is required'}), 400

    try:
        image = Image.open(image_file).convert('L')
        image_tensor = transform(image).unsqueeze(0)
        flat_image = image_tensor.view(-1).numpy().reshape(1, -1)
    except Exception as e:
        return jsonify({'error': f'Image preprocessing failed: {str(e)}'}), 400

    try:
        model, model_type = load_model(model_name)
        if model_type == 'torch':
            with torch.no_grad():
                output = model(image_tensor)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()
        elif model_type == 'sklearn':
            prediction = model.predict(flat_image)[0]
        else:
            return jsonify({'error': 'Unsupported model type'}), 500
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

    return jsonify({
        'model': model_name,
        'prediction': int(prediction)
    })

# ------------------- Start Server -------------------
if __name__ == '__main__':
    app.run(debug=True)
