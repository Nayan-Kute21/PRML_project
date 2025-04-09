"""
Model training functions for the leaf classification project.
"""

import os
import numpy as np
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

def train_knn_model(X_train, y_train, n_neighbors=5, cv=5):
    """
    Train a K-Nearest Neighbors model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    n_neighbors : int, optional
        Number of neighbors
    cv : int, optional
        Number of cross-validation folds for hyperparameter tuning
        
    Returns:
    --------
    knn_model : sklearn.neighbors.KNeighborsClassifier
        Trained KNN model
    """
    print("\nTraining KNN model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/knn', exist_ok=True)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    
    # Initialize the model
    knn = KNeighborsClassifier()
    
    # Perform grid search
    grid_search = GridSearchCV(knn, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    knn_model = grid_search.best_estimator_
    
    print(f"Best KNN parameters: {grid_search.best_params_}")
    print(f"KNN CV score: {grid_search.best_score_:.4f}")
    
    # Save the model
    joblib.dump(knn_model, 'models/knn/knn_model.pkl')
    
    return knn_model

def train_svm_model(X_train, y_train, cv=5):
    """
    Train a Support Vector Machine model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    cv : int, optional
        Number of cross-validation folds for hyperparameter tuning
        
    Returns:
    --------
    svm_model : sklearn.svm.SVC
        Trained SVM model
    """
    print("\nTraining SVM model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/svm', exist_ok=True)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto', 0.01, 0.1]
    }
    
    # Initialize the model
    svm = SVC(probability=True, random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(svm, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    svm_model = grid_search.best_estimator_
    
    print(f"Best SVM parameters: {grid_search.best_params_}")
    print(f"SVM CV score: {grid_search.best_score_:.4f}")
    
    # Save the model
    joblib.dump(svm_model, 'models/svm/svm_model.pkl')
    
    return svm_model

def train_decision_tree_model(X_train, y_train, cv=5):
    """
    Train a Decision Tree model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    cv : int, optional
        Number of cross-validation folds for hyperparameter tuning
        
    Returns:
    --------
    dt_model : sklearn.tree.DecisionTreeClassifier
        Trained Decision Tree model
    """
    print("\nTraining Decision Tree model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/decision_tree', exist_ok=True)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['gini', 'entropy']
    }
    
    # Initialize the model
    dt = DecisionTreeClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(dt, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    dt_model = grid_search.best_estimator_
    
    print(f"Best Decision Tree parameters: {grid_search.best_params_}")
    print(f"Decision Tree CV score: {grid_search.best_score_:.4f}")
    
    # Save the model
    joblib.dump(dt_model, 'models/decision_tree/dt_model.pkl')
    
    return dt_model

def train_random_forest_model(X_train, y_train, cv=5):
    """
    Train a Random Forest model.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    cv : int, optional
        Number of cross-validation folds for hyperparameter tuning
        
    Returns:
    --------
    rf_model : sklearn.ensemble.RandomForestClassifier
        Trained Random Forest model
    """
    print("\nTraining Random Forest model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/random_forest', exist_ok=True)
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize the model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(rf, param_grid, cv=cv, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    rf_model = grid_search.best_estimator_
    
    print(f"Best Random Forest parameters: {grid_search.best_params_}")
    print(f"Random Forest CV score: {grid_search.best_score_:.4f}")
    
    # Save the model
    joblib.dump(rf_model, 'models/random_forest/rf_model.pkl')
    
    return rf_model

def train_ann_model(X_train, y_train, X_val, y_val, n_classes):
    """
    Train an Artificial Neural Network model.
    
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
    n_classes : int
        Number of classes
        
    Returns:
    --------
    ann_model : tensorflow.keras.models.Sequential
        Trained ANN model
    history : tensorflow.keras.callbacks.History
        Training history
    """
    print("\nTraining ANN model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/ann', exist_ok=True)
    
    # One-hot encode the target variable for neural networks
    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=n_classes)
    
    # Define the ANN model
    ann_model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])
    
    ann_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Training the model
    history = ann_model.fit(
        X_train, y_train_one_hot,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val_one_hot),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    _, ann_accuracy = ann_model.evaluate(X_val, y_val_one_hot, verbose=0)
    print(f"ANN Accuracy: {ann_accuracy:.4f}")
    
    # Save ANN model
    ann_model.save('models/ann/ann_model')
    
    return ann_model, history

def train_cnn_model(X_train_img, y_train, X_val_img, y_val, n_classes):
    """
    Train a Convolutional Neural Network model.
    
    Parameters:
    -----------
    X_train_img : numpy.ndarray
        Training image data
    y_train : numpy.ndarray
        Training labels
    X_val_img : numpy.ndarray
        Validation image data
    y_val : numpy.ndarray
        Validation labels
    n_classes : int
        Number of classes
        
    Returns:
    --------
    cnn_model : tensorflow.keras.models.Sequential
        Trained CNN model
    history : tensorflow.keras.callbacks.History
        Training history
    """
    print("\nTraining CNN model...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models/cnn', exist_ok=True)
    
    # One-hot encode the target variable
    y_train_one_hot = to_categorical(y_train, num_classes=n_classes)
    y_val_one_hot = to_categorical(y_val, num_classes=n_classes)
    
    # Define the CNN model
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=X_train_img.shape[1:]),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(n_classes, activation='softmax')
    ])
    
    cnn_model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Training the model
    history = cnn_model.fit(
        X_train_img, y_train_one_hot,
        epochs=30,
        batch_size=16,
        validation_data=(X_val_img, y_val_one_hot),
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate the model
    _, cnn_accuracy = cnn_model.evaluate(X_val_img, y_val_one_hot, verbose=0)
    print(f"CNN Accuracy: {cnn_accuracy:.4f}")
    
    # Save CNN model
    cnn_model.save('models/cnn/cnn_model')
    
    return cnn_model, history