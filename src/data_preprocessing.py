"""
Data preprocessing utilities for the leaf classification project.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

def load_data(train_path='data/raw/train.csv', test_path='data/raw/test.csv'):
    """
    Load train and test data from CSV files.
    
    Parameters:
    -----------
    train_path : str
        Path to the training CSV file
    test_path : str
        Path to the test CSV file
        
    Returns:
    --------
    X_train : pandas.DataFrame
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : pandas.DataFrame
        Test features
    test_ids : pandas.Series
        Test IDs
    classes : numpy.ndarray
        Class names
    """
    print("Loading data...")
    
    # Ensure data directories exist
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Extract the target variable (species)
    species = train_df['species']
    
    # Drop the species column and id column from the training data
    X_train = train_df.drop(['species', 'id'], axis=1)
    
    # Extract ids from test data
    test_ids = test_df['id']
    X_test = test_df.drop(['id'], axis=1)
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(species)
    
    # Save the label encoder for later use
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Number of classes: {len(label_encoder.classes_)}")
    
    return X_train, y_train, X_test, test_ids, label_encoder.classes_

def preprocess_data(X_train, y_train, X_test=None, test_size=0.2):
    """
    Preprocess the data including scaling and splitting.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame
        Training features
    y_train : numpy.ndarray
        Training labels
    X_test : pandas.DataFrame, optional
        Test features
    test_size : float, optional
        Proportion of the dataset to be used for validation
        
    Returns:
    --------
    X_train_split : pandas.DataFrame
        Training features (if X_test is None) or scaled training features
    X_val : pandas.DataFrame
        Validation features (if X_test is None) or None
    y_train_split : numpy.ndarray
        Training labels (if X_test is None) or all training labels
    y_val : numpy.ndarray
        Validation labels (if X_test is None) or None
    shape_cols : list
        Column names for shape features
    margin_cols : list
        Column names for margin features
    texture_cols : list
        Column names for texture features
    """
    print("Preprocessing data...")
    
    # Split features into shape, margin, and texture
    shape_cols = [col for col in X_train.columns if 'shape' in col]
    margin_cols = [col for col in X_train.columns if 'margin' in col]
    texture_cols = [col for col in X_train.columns if 'texture' in col]
    
    print(f"Number of shape features: {len(shape_cols)}")
    print(f"Number of margin features: {len(margin_cols)}")
    print(f"Number of texture features: {len(texture_cols)}")
    
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    
    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    
    # Split data for training and validation
    if X_test is None:
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train_scaled, y_train, test_size=test_size, random_state=42, stratify=y_train
        )
        return X_train_split, X_val, y_train_split, y_val, shape_cols, margin_cols, texture_cols
    else:
        # If test data is provided, scale it using the same scaler
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        return X_train_scaled, y_train, X_test_scaled, shape_cols, margin_cols, texture_cols

def load_images(image_path, ids, img_size=(224, 224)):
    """
    Load and preprocess images for CNN model.
    
    Parameters:
    -----------
    image_path : str
        Path to the directory containing images
    ids : list
        List of image IDs to load
    img_size : tuple, optional
        Size to resize images to
        
    Returns:
    --------
    images : numpy.ndarray
        Array of preprocessed images
    """
    from PIL import Image
    import glob
    
    images = []
    for id_val in ids:
        # Find the image file with the given ID
        img_files = glob.glob(f"{image_path}/{id_val}.*")
        if img_files:
            img_file = img_files[0]
            img = Image.open(img_file).convert('RGB').resize(img_size)
            img_array = np.array(img) / 255.0  # Normalize
            images.append(img_array)
        else:
            print(f"Warning: No image found for ID {id_val}")
    
    return np.array(images)