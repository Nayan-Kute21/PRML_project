"""
Feature extraction utilities for the leaf classification project.
"""

import joblib
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def apply_dimension_reduction(X_train, y_train, X_val=None, method='pca', n_components=20):
    """
    Apply dimension reduction techniques to the data.
    
    Parameters:
    -----------
    X_train : pandas.DataFrame or numpy.ndarray
        Training features
    y_train : numpy.ndarray
        Training labels
    X_val : pandas.DataFrame or numpy.ndarray, optional
        Validation features
    method : str, optional
        Dimension reduction method ('pca' or 'lda')
    n_components : int, optional
        Number of components to keep
        
    Returns:
    --------
    X_train_reduced : numpy.ndarray
        Reduced training features
    X_val_reduced : numpy.ndarray or None
        Reduced validation features (if X_val is not None)
    reducer : sklearn.decomposition.PCA or sklearn.discriminant_analysis.LinearDiscriminantAnalysis
        Fitted dimension reduction model
    """
    print(f"Applying {method.upper()} for dimension reduction...")
    
    if method.lower() == 'pca':
        reducer = PCA(n_components=n_components)
        X_train_reduced = reducer.fit_transform(X_train)
        explained_variance = reducer.explained_variance_ratio_.sum()
        print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
    
    elif method.lower() == 'lda':
        # LDA components cannot exceed number of classes minus 1
        n_components_lda = min(n_components, len(set(y_train)) - 1)
        reducer = LDA(n_components=n_components_lda)
        X_train_reduced = reducer.fit_transform(X_train, y_train)
        print(f"LDA reduced dimensionality to {n_components_lda} components")
    
    else:
        raise ValueError("Method must be either 'pca' or 'lda'")
    
    # Save the reducer
    joblib.dump(reducer, f'models/{method}_reducer.pkl')
    
    if X_val is not None:
        X_val_reduced = reducer.transform(X_val)
        return X_train_reduced, X_val_reduced, reducer
    else:
        return X_train_reduced, reducer

def get_feature_groups(X):
    """
    Split features into shape, margin, and texture groups.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature DataFrame
        
    Returns:
    --------
    shape_features : pandas.DataFrame
        Shape features
    margin_features : pandas.DataFrame
        Margin features
    texture_features : pandas.DataFrame
        Texture features
    """
    shape_cols = [col for col in X.columns if 'shape' in col]
    margin_cols = [col for col in X.columns if 'margin' in col]
    texture_cols = [col for col in X.columns if 'texture' in col]
    
    shape_features = X[shape_cols]
    margin_features = X[margin_cols]
    texture_features = X[texture_cols]
    
    return shape_features, margin_features, texture_features