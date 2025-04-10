from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
import tensorflow as tf
import os
import pandas as pd
from werkzeug.utils import secure_filename
from skimage.feature import local_binary_pattern
from pyefd import elliptic_fourier_descriptors
from flask_cors import CORS 
app = Flask(__name__)
CORS(app)
# Load all models and preprocessing tools
def load_models():
    models = {}
    preprocessors = {}
    
    # Load traditional ML models
    model_paths = {
        'DT': 'models/decision_tree/dt_model.pkl', 
        'KNN': 'models/knn/knn_model.pkl',
        'RF': 'models/random_forest/rf_model.pkl',
        'SVM': 'models/svm/svm_model.pkl',
        'ANN': 'models/ann/ann_model/saved_model.pb',
    }
    
    for model_name, file_path in model_paths.items():
        try:
            models[model_name] = joblib.load(file_path)
            print(f"Successfully loaded {model_name} model")
        except Exception as e:
            print(f"Error loading {model_name} model: {e}")
    
    # Load ANN model
    try:
        models['ANN'] = tf.keras.models.load_model('models/ann/ann_model')
        print("Successfully loaded ANN model")
    except Exception as e:
        print(f"Error loading ANN model: {e}")
    
    # # Try to load CNN model
    # try:
    #     models['CNN'] = tf.keras.models.load_model('models/cnn/cnn_model')
    #     print("Successfully loaded CNN model")
    # except Exception as e:
    #     print(f"Error loading CNN model: {e}")
    
    # Load preprocessing tools
    preproc_files = {
        'scaler': 'models/scaler.pkl',
        'label_encoder': 'models/label_encoder.pkl'
    }
    
    # Try to load dimension reduction model (PCA or LDA)
    if os.path.exists('models/pca_reducer.pkl'):
        preproc_files['reducer'] = 'models/pca_reducer.pkl'
    elif os.path.exists('models/lda_reducer.pkl'):
        preproc_files['reducer'] = 'models/lda_reducer.pkl'
    
    for name, file_path in preproc_files.items():
        try:
            preprocessors[name] = joblib.load(file_path)
            print(f"Successfully loaded {name}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    
    return models, preprocessors

# Create uploads directory if it doesn't exist
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def extract_shape_features(image, n_descriptors=16):
    """
    Extract shape features using elliptic Fourier descriptors.
    """
    # Find contours in the binary image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(64)  # Return zeros if no contours found
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Make sure the contour has enough points
    if len(largest_contour) < 3:
        return np.zeros(64)
    
    # Calculate elliptic Fourier descriptors
    coeffs = elliptic_fourier_descriptors(np.squeeze(largest_contour), order=n_descriptors, normalize=True)
    
    # Flatten and take first 64 components
    return coeffs.flatten()[:64]  # Shape: (64,)

def extract_texture_features(image, P=8, R=1):
    """
    Extract texture features using Local Binary Pattern (LBP).
    """
    lbp = local_binary_pattern(image, P, R, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2), density=True)
    
    # Ensure we have exactly 64 texture features
    if len(hist) < 64:
        hist = np.pad(hist, (0, 64 - len(hist)), 'constant')
    return hist[:64]

def extract_margin_features(image, n_points=64):
    """
    Extract margin features based on distance from centroid along the contour.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.zeros(64)  # Return zeros if no contours found
    
    cnt = max(contours, key=cv2.contourArea)
    
    if cv2.contourArea(cnt) == 0:
        return np.zeros(64)  # Return zeros if contour area is zero
    
    M = cv2.moments(cnt)
    
    if M['m00'] == 0:
        return np.zeros(64)  # Return zeros if moment is zero
    
    cx, cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    
    # Calculate distances from centroid to perimeter points
    distances = []
    for theta in np.linspace(0, 2*np.pi, n_points, endpoint=False):
        # Create a ray from centroid
        ray_end_x = cx + np.cos(theta) * 1000
        ray_end_y = cy + np.sin(theta) * 1000
        
        # Find intersection with contour (distance to boundary)
        dist = cv2.pointPolygonTest(cnt, (ray_end_x, ray_end_y), True)
        distances.append(abs(dist))
    
    distances = np.array(distances)
    
    # Normalize distances
    if distances.max() != 0:
        distances = distances / distances.max()
    
    return distances

def preprocess_image(image_path, image_size=(224, 224)):
    """
    Preprocess the image and extract features for model input
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    # Convert to grayscale if not already
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Resize for consistent processing
    gray_resized = cv2.resize(gray, (256, 256))
    
    # Apply threshold to make it binary
    _, binary_img = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract features
    shape_feat = extract_shape_features(binary_img)
    texture_feat = extract_texture_features(gray_resized)
    margin_feat = extract_margin_features(binary_img)

    # Create feature dictionary
    feature_dict = {}
    
    # Add margin features
    for i in range(len(margin_feat)):
        feature_dict[f'margin{i+1}'] = margin_feat[i]
    
    # Add shape features
    for i in range(len(shape_feat)):
        feature_dict[f'shape{i+1}'] = shape_feat[i]
    
    # Add texture features
    for i in range(len(texture_feat)):
        feature_dict[f'texture{i+1}'] = texture_feat[i]
    
    # Convert to DataFrame to maintain column order
    features_df = pd.DataFrame([feature_dict])
    
    # Prepare image for CNN (if needed)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, image_size)
    img_normalized = img_resized / 255.0
    
    return features_df, np.expand_dims(img_normalized, axis=0)

# Load all models and preprocessors at startup
models, preprocessors = load_models()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    if 'model' not in request.values:
        return jsonify({'error': 'No model name provided'}), 400
    
    model_name = request.values['model'].upper()
    
    if model_name not in models:
        return jsonify({'error': f'Invalid model name. Choose from: {", ".join(models.keys())}'}), 400
    
    # Save the uploaded image temporarily
    image = request.files.get('image')
    image_path = "temp_image.jpg"
    image.save(image_path)
    
    
    try:
        # Preprocess the image
        features_df, img_array = preprocess_image(image_path)
        print(f"Image features extracted: {features_df.columns.tolist()}")
        print(f"Image shape for CNN: {img_array.shape}")
        if features_df is None:
            return jsonify({'error': 'Could not process image'}), 400
        
        # Apply the same preprocessing steps as during training
        if model_name != 'CNN':
            # Apply scaler
            if 'scaler' in preprocessors:
                features_scaled = preprocessors['scaler'].transform(features_df)
                features_df = pd.DataFrame(features_scaled, columns=features_df.columns)
                print(f"Scaled features: {features_scaled}")
                print(f"Scaled features shape: {features_df.shape}")
            # Apply dimension reduction if available
            if 'reducer' in preprocessors:
                reduced_features = preprocessors['reducer'].transform(features_df)
                print(f"Reduced features: {reduced_features}")
                # For neural network, we need the reduced features
                if model_name == 'ANN':
                    prediction = models[model_name].predict(reduced_features)
                    pred_class = np.argmax(prediction, axis=1)[0]
                else:
                    # For traditional ML models
                    prediction = models[model_name].predict(reduced_features)
                    print(f"Prediction: {prediction}")
                    pred_class = prediction[0]
                    print(f"Prediction class: {pred_class}")
            else:
                # If no reducer is found, use the scaled features
                if model_name == 'ANN':
                    prediction = models[model_name].predict(features_df)
                    pred_class = np.argmax(prediction, axis=1)[0]
                else:
                    prediction = models[model_name].predict(features_df)
                    print(f"Prediction: {prediction}")
                    pred_class = prediction[0]
        else:
            # For CNN, use the image array directly
            prediction = models[model_name].predict(img_array)
            pred_class = np.argmax(prediction, axis=1)[0]
        
        # Convert prediction to human-readable label if label encoder is available
        if 'label_encoder' in preprocessors:
            try:
                pred_label = preprocessors['label_encoder'].inverse_transform([pred_class])[0]
            except:
                pred_label = str(pred_class)
        else:
            pred_label = str(pred_class)
        
        # Clean up - remove the temporary file
        
        
        # If model provides probabilities, include them
        probabilities = None
        if hasattr(models[model_name], 'predict_proba') and model_name not in ['ANN', 'CNN']:
            proba = models[model_name].predict_proba(reduced_features if 'reducer' in preprocessors else features_df)
            probabilities = {str(i): float(p) for i, p in enumerate(proba[0])}
        elif model_name in ['ANN', 'CNN']:
            # For neural networks, prediction already contains probabilities
            probabilities = {str(i): float(p) for i, p in enumerate(prediction[0])}
        
        response = {
            'success': True,
            'model_used': model_name,
            'prediction_class': int(pred_class) if isinstance(pred_class, (np.integer, int)) else pred_class,
            'prediction_label': pred_label
        }
        print(response)
        
        if probabilities:
            response['probabilities'] = probabilities
            
        return jsonify(response)
        
    except Exception as e:
        # Clean up on error
        try:
            os.remove(image_path)
        except:
            pass
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint
    """
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'preprocessors_loaded': list(preprocessors.keys())
    })

if __name__ == '__main__':
    app.run(debug=True)