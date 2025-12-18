# High-level stacking pipeline
"""
Stacking ensemble pipeline utilities.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict


def create_meta_features(base_models, X, y=None, cv=5):
    """
    Create meta-features using cross-validation predictions from base models.
    
    Args:
        base_models: List of (name, model) tuples
        X: Features
        y: Labels (for training, None for inference)
        cv: Number of cross-validation folds
    
    Returns:
        np.ndarray: Meta-features (stacked predictions)
    """
    meta_features = []
    
    for name, model in base_models:
        print(f"Generating meta-features from {name}...")
        
        if y is not None:
            # Training: use cross-val predictions to avoid overfitting
            if hasattr(model, 'predict_proba'):
                preds = cross_val_predict(model, X, y, cv=cv, method='predict_proba')
            else:
                preds = cross_val_predict(model, X, y, cv=cv)
                if len(preds.shape) == 1:
                    preds = preds.reshape(-1, 1)
        else:
            # Inference: direct predictions
            if hasattr(model, 'predict_proba'):
                preds = model.predict_proba(X)
            else:
                preds = model.predict(X)
                if len(preds.shape) == 1:
                    preds = preds.reshape(-1, 1)
        
        meta_features.append(preds)
    
    return np.hstack(meta_features)


def build_meta_features(*predictions):
    """
    Build meta-features by horizontally stacking base model predictions.
    
    Args:
        *predictions: Variable number of prediction arrays (predictions or probabilities)
        
    Returns:
        np.ndarray: Stacked meta-features
    """
    processed_preds = []
    
    for pred in predictions:
        if len(pred.shape) == 1:
            # Single prediction array, reshape to column vector
            processed_preds.append(pred.reshape(-1, 1))
        else:
            # Probability matrix or multi-dimensional predictions
            processed_preds.append(pred)
    
    # Stack horizontally to create meta-features
    meta_features = np.hstack(processed_preds)
    
    return meta_features


def evaluate_ensemble(meta_learner, X_meta, y_true, model_names=None):
    """
    Evaluate ensemble performance on meta-features.
    
    Args:
        meta_learner: Trained meta-learner model
        X_meta: Meta-features from base models
        y_true: True labels
        model_names: Names of base models (optional)
    
    Returns:
        dict: Evaluation metrics
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    # Get predictions
    y_pred = meta_learner.predict(X_meta)
    
    # Get probabilities if available
    if hasattr(meta_learner, 'predict_proba'):
        y_prob = meta_learner.predict_proba(X_meta)
        if len(y_prob.shape) > 1:
            y_prob = y_prob[:, 1]  # Probability of positive class
    else:
        y_prob = None
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['roc_auc'] = None
    
    return metrics


def get_feature_importance(meta_learner, model_names):
    """
    Extract feature importance from meta-learner.
    
    Args:
        meta_learner: Trained meta-learner model
        model_names: Names of base models
    
    Returns:
        pd.DataFrame: Feature importance by model
    """
    if hasattr(meta_learner, 'feature_importances_'):
        # Tree-based models (XGBoost, RandomForest)
        importance = meta_learner.feature_importances_
    elif hasattr(meta_learner, 'coef_'):
        # Linear models (LogisticRegression, SVM)
        importance = np.abs(meta_learner.coef_[0])
    else:
        print("⚠️ Meta-learner doesn't have feature importance")
        return None
    
    # Create DataFrame
    df = pd.DataFrame({
        'model': model_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    return df


def combine_predictions(predictions_list, method='average'):
    """
    Combine predictions from multiple models.
    
    Args:
        predictions_list: List of prediction arrays
        method: Combination method ('average', 'voting', 'weighted')
    
    Returns:
        np.ndarray: Combined predictions
    """
    if method == 'average':
        # Average probabilities
        return np.mean(predictions_list, axis=0)
    elif method == 'voting':
        # Majority voting on hard predictions
        stacked = np.stack(predictions_list, axis=0)
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=stacked)
    else:
        raise ValueError(f"Unknown method: {method}")
