"""
Main script for the leaf classification project.
"""

import os
import numpy as np
import pandas as pd
import argparse
import joblib
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Import project modules
from data_preprocessing import load_data, preprocess_data, load_images
from feature_extraction import apply_dimension_reduction
from model_training import (
    train_knn_model, train_svm_model, train_decision_tree_model, 
    train_random_forest_model, train_ann_model, train_cnn_model
)
from evaluation import (
    evaluate_model, compare_models, analyze_failures, generate_submission
)

def create_directories():
    """Create all necessary directories for the project."""
    directories = [
        'data/raw',
        'data/processed',
        'models/knn',
        'models/svm',
        'models/decision_tree',
        'models/random_forest',
        'models/ann',
        'models/cnn',
        'results/metrics',
        'results/visualizations',
        'results/failure_analysis'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def run_traditional_models(X_train, y_train, X_val, y_val, classes, include_models=None):
    """
    Train and evaluate traditional machine learning models.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels
    classes : numpy.ndarray
        Class names
    include_models : list, optional
        List of models to include. If None, all models are included.
        
    Returns:
    --------
    results : dict
        Dictionary of model results
    """
    results = {}
    
    # Define available models
    model_functions = {
        'knn': train_knn_model,
        'svm': train_svm_model,
        'decision_tree': train_decision_tree_model,
        'random_forest': train_random_forest_model
    }
    
    # If no specific models are requested, use all
    if include_models is None:
        include_models = list(model_functions.keys())
    
    # Train and evaluate each selected model
    for model_name in include_models:
        if model_name.lower() in model_functions:
            print(f"\n{'='*50}")
            print(f"Processing {model_name.upper()} model")
            print(f"{'='*50}")
            
            # Train the model
            model = model_functions[model_name.lower()](X_train, y_train)
            
            # Evaluate the model
            accuracy, predictions = evaluate_model(model, X_val, y_val, model_name.upper(), classes)
            
            # Store results
            results[model_name.upper()] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'model': model
            }
        else:
            print(f"Warning: Unknown model '{model_name}'. Skipping.")
    
    return results

def run_neural_network_models(X_train, y_train, X_val, y_val, X_train_img=None, X_val_img=None, classes=None, include_models=None):
    results = {}
    n_classes = len(set(y_train))
    
    # Define available models
    available_models = ['ann']
    if X_train_img is not None and X_val_img is not None:
        available_models.append('cnn')
    
    # If no specific models are requested, use all available
    if include_models is None:
        include_models = available_models
    else:
        # Convert model names to lowercase for case-insensitive comparison
        include_models = [model.lower() for model in include_models]
        # Filter to only include available models
        include_models = [model for model in include_models if model in available_models]
    
    for model_name in include_models:
        if model_name.lower() == 'ann':
            print(f"\n{'='*50}")
            print(f"Processing ANN model")
            print(f"{'='*50}")
            
            # Train ANN model
            ann_model, history = train_ann_model(X_train, y_train, X_val, y_val, n_classes)
            
            # Evaluate the model
            accuracy, predictions = evaluate_model(ann_model, X_val, y_val, 'ANN', classes)
            
            # Store results
            results['ANN'] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'history': history.history,
                'model': ann_model
            }
        
        elif model_name.lower() == 'cnn' and X_train_img is not None and X_val_img is not None:
            print(f"\n{'='*50}")
            print(f"Processing CNN model")
            print(f"{'='*50}")
            
            # Train CNN model
            cnn_model, history = train_cnn_model(X_train_img, y_train, X_val_img, y_val, n_classes)
            
            # Evaluate the model
            accuracy, predictions = evaluate_model(cnn_model, X_val_img, y_val, 'CNN', classes)
            
            # Store results
            results['CNN'] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'history': history.history,
                'model': cnn_model
            }
    
    return results


def main(args):
    """
    Main function to run the leaf classification pipeline.
    
    Parameters:
    -----------
    args : argparse.Namespace
        Command-line arguments
    """
    # Create necessary directories
    create_directories()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print(f"\n{'='*50}")
    print("LEAF CLASSIFICATION PROJECT")
    print(f"{'='*50}\n")
    
    # Step 1: Load data
    print(f"\n{'='*50}")
    print("Step 1: Loading Data")
    print(f"{'='*50}")
    
    X_train, y_train, X_test, test_ids, classes = load_data(
        train_path=args.train_path,
        test_path=args.test_path
    )
    
    # Step 2: Preprocess data
    print(f"\n{'='*50}")
    print("Step 2: Preprocessing Data")
    print(f"{'='*50}")
    
    X_train_split, X_val, y_train_split, y_val, shape_cols, margin_cols, texture_cols = preprocess_data(
        X_train, y_train, test_size=args.test_size
    )
    
    # Step 3: Feature engineering with dimension reduction
    print(f"\n{'='*50}")
    print("Step 3: Feature Engineering")
    print(f"{'='*50}")
    
    X_train_reduced, X_val_reduced, _ = apply_dimension_reduction(
        X_train_split, y_train_split, X_val, 
        method=args.dim_reduction, 
        n_components=args.n_components
    )
    
    # Step 4: Load images if needed for CNN
    X_train_img, X_val_img = None, None

    if any(model.lower() == 'cnn' for model in args.models) and os.path.exists(args.images_path):
        print(f"\n{'='*50}")
        print("Step 4: Loading Images for CNN")
        print(f"{'='*50}")
        
        # Create train/validation split for images that matches the other data split
        from sklearn.model_selection import train_test_split
        
        # Get all training data IDs
        train_df = pd.read_csv(args.train_path)
        all_train_ids = train_df['id'].values
        
        # Split IDs the same way we split the other data
        train_indices, val_indices = train_test_split(
            np.arange(len(all_train_ids)), 
            test_size=args.test_size, 
            random_state=42, 
            stratify=y_train
        )
        
        # Get IDs for training and validation sets
        train_ids = all_train_ids[train_indices]
        val_ids = all_train_ids[val_indices]
        
        # Load images
        X_train_img = load_images(args.images_path, train_ids)
        X_val_img = load_images(args.images_path, val_ids)
        
        if X_train_img is not None and len(X_train_img) > 0 and X_val_img is not None and len(X_val_img) > 0:
            print(f"Successfully loaded {len(X_train_img)} training images and {len(X_val_img)} validation images")
        else:
            print("Failed to load images or no images found. CNN model will be skipped.")
            X_train_img, X_val_img = None, None
    
    # Step 5: Train and evaluate traditional models
    print(f"\n{'='*50}")
    print("Step 5: Training and Evaluating Models")
    print(f"{'='*50}")
    
    traditional_models = [m for m in args.models if m.lower() not in ['ann', 'cnn']]
    nn_models = [m for m in args.models if m.lower() in ['ann', 'cnn']]
    
    # Train traditional ML models
    results = run_traditional_models(
        X_train_reduced, y_train_split, X_val_reduced, y_val,
        classes, include_models=traditional_models
    )
    
    # Train neural network models
    nn_results = run_neural_network_models(
        X_train_reduced, y_train_split, X_val_reduced, y_val,
        X_train_img, X_val_img, classes, include_models=nn_models
    )
    
    # Combine results
    results.update(nn_results)
    
    # Step 6: Compare models and analyze failures
    print(f"\n{'='*50}")
    print("Step 6: Model Comparison and Analysis")
    print(f"{'='*50}")
    
    best_model_name, accuracies = compare_models(results, y_val, classes)
    
    analyze_failures(
        results, best_model_name, 
        X_val_reduced if best_model_name.lower() not in ['cnn'] else X_val_img,
        y_val, classes
    )
    
    # Step 7: Generate submission with the best model
    print(f"\n{'='*50}")
    print("Step 7: Generating Submission")
    print(f"{'='*50}")
    
    # Load the label encoder
    label_encoder = joblib.load('models/label_encoder.pkl')
    
    # Process test data based on the best model
    best_model = results[best_model_name]['model']
    
    if best_model_name.lower() == 'cnn':
        # For CNN, we need to handle image test data
        if os.path.exists(args.images_path):
            # Load test images
            test_df = pd.read_csv(args.test_path)
            test_ids = test_df['id'].values
            X_test_img = load_images(args.images_path, test_ids)
            
            if X_test_img is not None:
                # Generate submission using CNN model and image test data
                generate_submission(best_model, X_test_img, test_ids, label_encoder, output_path=args.submission_path)
            else:
                print("Failed to load test images. Using the next best model instead.")
                # Find the next best model
                models_by_accuracy = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
                for model_name, _ in models_by_accuracy:
                    if model_name.lower() != 'cnn':
                        next_best_model_name = model_name
                        next_best_model = results[next_best_model_name]['model']
                        
                        # Process test data for next best model
                        X_test_scaled = joblib.load('models/scaler.pkl').transform(X_test)
                        if args.dim_reduction.lower() == 'pca':
                            pca_reducer = joblib.load('models/pca_reducer.pkl')
                            X_test_reduced = pca_reducer.transform(X_test_scaled)
                        else:
                            lda_reducer = joblib.load('models/lda_reducer.pkl')
                            X_test_reduced = lda_reducer.transform(X_test_scaled)
                        
                        # Generate submission using next best model
                        print(f"Using {next_best_model_name} as the fallback model.")
                        generate_submission(next_best_model, X_test_reduced, test_ids, label_encoder, output_path=args.submission_path)
                        break
        else:
            print("Image data path not found. Using the next best model instead.")
            # Similar fallback code as above
    else:
        # For non-CNN models, process test data as usual
        X_test_scaled = joblib.load('models/scaler.pkl').transform(X_test)
        
        if args.dim_reduction.lower() == 'pca':
            pca_reducer = joblib.load('models/pca_reducer.pkl')
            X_test_reduced = pca_reducer.transform(X_test_scaled)
        else:
            lda_reducer = joblib.load('models/lda_reducer.pkl')
            X_test_reduced = lda_reducer.transform(X_test_scaled)
        
        # Generate submission
        generate_submission(best_model, X_test_reduced, test_ids, label_encoder, output_path=args.submission_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leaf Classification Project")
    
    parser.add_argument('--train_path', type=str, default='data/raw/train.csv',
                        help='Path to the training CSV file')
    parser.add_argument('--test_path', type=str, default='data/raw/test.csv',
                        help='Path to the test CSV file')
    parser.add_argument('--images_path', type=str, default='data/raw/images',
                        help='Path to the images directory')
    parser.add_argument('--submission_path', type=str, default='data/processed/submission.csv',
                        help='Path to save the submission file')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of the dataset to be used for validation')
    parser.add_argument('--dim_reduction', type=str, default='pca', choices=['pca', 'lda'],
                        help='Dimension reduction method to use')
    parser.add_argument('--n_components', type=int, default=30,
                        help='Number of components for dimension reduction')
    parser.add_argument('--models', nargs='+', default=['knn', 'svm', 'decision_tree', 'random_forest', 'ann'],
                        help='Models to train and evaluate')
    
    args = parser.parse_args()
    main(args)