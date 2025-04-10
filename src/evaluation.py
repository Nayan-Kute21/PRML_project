"""
Model evaluation functions for the leaf classification project.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model

def evaluate_model(model, X_val, y_val, model_name, classes=None):
    """
    Evaluate a trained model.
    
    Parameters:
    -----------
    model : sklearn model or tensorflow.keras.models.Model
        Trained model
    X_val : numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels
    model_name : str
        Name of the model
    classes : numpy.ndarray, optional
        Class names
        
    Returns:
    --------
    accuracy : float
        Model accuracy
    predictions : numpy.ndarray
        Model predictions
    """
    print(f"\nEvaluating {model_name} model...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results/metrics', exist_ok=True)
    
    # Make predictions
    if model_name.lower() == 'ann' or model_name.lower() == 'cnn':
        # For neural network models
        from tensorflow.keras.utils import to_categorical
        y_val_one_hot = to_categorical(y_val)
        y_pred_proba = model.predict(X_val)
        predictions = np.argmax(y_pred_proba, axis=1)
        accuracy = accuracy_score(y_val, predictions)
    else:
        # For sklearn models
        predictions = model.predict(X_val)
        accuracy = accuracy_score(y_val, predictions)
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    
    # Classification report
    if classes is not None:
        report = classification_report(y_val, predictions, target_names=classes)
    else:
        report = classification_report(y_val, predictions)
    
    print(f"\nClassification Report for {model_name}:")
    print(report)
    
    # Save the classification report to a file
    with open(f'results/metrics/{model_name.lower()}_classification_report.txt', 'w') as f:
        f.write(report)
    
    return accuracy, predictions

def compare_models(results, y_val, classes=None):
    """
    Compare the performance of different models.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    y_val : numpy.ndarray
        Validation labels
    classes : numpy.ndarray, optional
        Class names
        
    Returns:
    --------
    best_model_name : str
        Name of the best-performing model
    accuracies : dict
        Dictionary of model accuracies
    """
    print("\nComparing model performances...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results/visualizations', exist_ok=True)
    
    # Accuracy comparison
    accuracies = {model: results[model]['accuracy'] for model in results if 'accuracy' in results[model]}
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig('results/visualizations/accuracy_comparison.png')
    plt.close()
    
    # Determine the best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = results[best_model_name]
    
    print(f"\nBest performing model: {best_model_name} with accuracy: {best_model['accuracy']:.4f}")
    
    # Create confusion matrix for the best model
    if 'predictions' in best_model:
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_val, best_model['predictions'])
        
        # If we have too many classes, don't show annotations
        annot = len(np.unique(y_val)) < 30
        
        sns.heatmap(cm, annot=annot, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {best_model_name}')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        plt.savefig('results/visualizations/confusion_matrix.png')
        plt.close()
    
    return best_model_name, accuracies

def analyze_failures(results, best_model_name, X_val, y_val, classes=None):
    """
    Analyze failure cases of the best model.
    
    Parameters:
    -----------
    results : dict
        Dictionary of model results
    best_model_name : str
        Name of the best-performing model
    X_val : pandas.DataFrame or numpy.ndarray
        Validation features
    y_val : numpy.ndarray
        Validation labels
    classes : numpy.ndarray, optional
        Class names
    """
    print("\nAnalyzing failure cases...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results/failure_analysis', exist_ok=True)
    
    best_model = results[best_model_name]
    
    if 'predictions' in best_model:
        predictions = best_model['predictions']
        
        # Find incorrect predictions
        incorrect_indices = np.where(predictions != y_val)[0]
        
        if len(incorrect_indices) > 0:
            print(f"Number of incorrect predictions: {len(incorrect_indices)}")
            
            # Analyze some of the misclassified examples
            num_examples = min(5, len(incorrect_indices))
            
            failure_analysis_report = []
            
            for i in range(num_examples):
                idx = incorrect_indices[i]
                
                if classes is not None:
                    true_class = classes[y_val[idx]]
                    pred_class = classes[predictions[idx]]
                else:
                    true_class = str(y_val[idx])
                    pred_class = str(predictions[idx])
                
                report = f"\nMisclassification {i+1}:\n"
                report += f"True class: {true_class}\n"
                report += f"Predicted class: {pred_class}\n"
                
                # For CNN models, we can't do typical feature importance analysis
                if best_model_name.lower() != 'cnn' and hasattr(X_val, 'iloc'):
                    feature_importance = analyze_feature_importance(best_model['model'], X_val.iloc[idx], classes)
                    
                    report += "\nFeature importance:\n"
                    for feature, importance in feature_importance:
                        report += f"{feature}: {importance}\n"
                
                failure_analysis_report.append(report)
                print(report)
                
                # Save the failure analysis
                with open(f'results/failure_analysis/misclassification_{i+1}.txt', 'w') as f:
                    f.write(report)
            
            # Save a combined report
            with open('results/failure_analysis/failure_analysis_summary.txt', 'w') as f:
                f.write(f"Failure Analysis for {best_model_name}\n")
                f.write(f"Total incorrect predictions: {len(incorrect_indices)} out of {len(y_val)}\n")
                f.write(f"Error rate: {len(incorrect_indices)/len(y_val):.2%}\n\n")
                for report in failure_analysis_report:
                    f.write(report + "\n")
        else:
            print("No misclassifications found in the validation set.")

def analyze_feature_importance(model, sample, classes=None):
    """
    Analyze feature importance for a specific prediction.
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    sample : pandas.Series or numpy.ndarray
        Single sample features
    classes : numpy.ndarray, optional
        Class names
        
    Returns:
    --------
    feature_importance : list
        List of (feature, importance) tuples
    """
    feature_importance = []
    
    # Handle different model types
    if hasattr(model, 'feature_importances_'):
        # For tree-based models (Random Forest, Decision Tree)
        feature_importances = model.feature_importances_
        if hasattr(sample, 'index'):
            # If sample is a pandas Series
            feature_names = sample.index
        else:
            # If sample is a numpy array
            feature_names = [f"feature_{i}" for i in range(len(sample))]
            
        for feature, importance in zip(feature_names, feature_importances):
            feature_importance.append((feature, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    elif hasattr(model, 'coef_'):
        # For linear models (SVM with linear kernel)
        if len(model.coef_.shape) == 2:
            # Multi-class case
            feature_importances = np.mean(np.abs(model.coef_), axis=0)
        else:
            # Binary case
            feature_importances = np.abs(model.coef_)
            
        if hasattr(sample, 'index'):
            # If sample is a pandas Series
            feature_names = sample.index
        else:
            # If sample is a numpy array
            feature_names = [f"feature_{i}" for i in range(len(sample))]
            
        for feature, importance in zip(feature_names, feature_importances):
            feature_importance.append((feature, importance))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    else:
        # For other models (KNN, SVM with non-linear kernel)
        # Approximate importance by perturbation
        if hasattr(sample, 'index'):
            # If sample is a pandas Series
            feature_names = sample.index
            sample_array = sample.values.reshape(1, -1)
        else:
            # If sample is a numpy array
            feature_names = [f"feature_{i}" for i in range(len(sample))]
            sample_array = sample.reshape(1, -1)
        
        # Get baseline prediction
        baseline_prediction = model.predict(sample_array)[0]
        
        # Test each feature's importance
        for i, feature in enumerate(feature_names):
            # Create a perturbed sample
            perturbed_sample = sample_array.copy()
            perturbed_sample[0, i] = 0  # Zero out the feature
            
            # Get prediction for perturbed sample
            perturbed_prediction = model.predict(perturbed_sample)[0]
            
            # Calculate importance as 1 if prediction changes, 0 otherwise
            importance = 1 if perturbed_prediction != baseline_prediction else 0
            feature_importance.append((feature, importance))
    
    # Return top 10 important features or all if less than 10
    return feature_importance[:min(10, len(feature_importance))]

def generate_submission(model, X_test, test_ids, label_encoder, output_path='data/processed/submission.csv'):
    """
    Generate submission file for Kaggle competition.
    
    Parameters:
    -----------
    model : sklearn model or tensorflow.keras.models.Model
        Trained model
    X_test : pandas.DataFrame or numpy.ndarray
        Test features
    test_ids : pandas.Series or numpy.ndarray
        Test IDs
    label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder used for training
    output_path : str, optional
        Path to save the submission file
    """
    print("\nGenerating submission file...")
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Make predictions
    if hasattr(model, 'predict_proba'):
        # For sklearn models with probability output
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    elif hasattr(model, 'layers'):
        # For Keras models (including CNN)
        from tensorflow.keras.utils import to_categorical
        
        # Get number of classes
        n_classes = len(label_encoder.classes_)
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # For other sklearn models
        y_pred = model.predict(X_test)
    
    # Convert numeric predictions back to species names
    y_pred_species = label_encoder.inverse_transform(y_pred)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'id': test_ids,
        'species': y_pred_species
    })
    
    # Save submission file
    submission.to_csv(output_path, index=False)
    print(f"Submission file saved to {output_path}")