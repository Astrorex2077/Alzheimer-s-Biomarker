# Level-1 meta-learner (stacking)
"""
Meta-Learner (Level-1 Stacking Model)

This module implements various meta-learners that combine
predictions from base models (Model A, B, C, D) to make
final predictions.

Supported meta-learners:
- Logistic Regression
- Multi-layer Perceptron (MLP)
- XGBoost
- PyTorch MLP

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib

from utils.config import (
    META_LEARNER_CONFIG,
    DEVICE,
    SAVED_MODELS_DIR,
    RANDOM_SEED,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# BASE META-LEARNER INTERFACE
# ============================================================================

class MetaLearner:
    """
    Base interface for meta-learners.
    
    All meta-learners should implement fit, predict, and predict_proba.
    """
    
    def fit(self, X_meta: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Fit meta-learner on meta-features.
        
        Args:
            X_meta: Meta-features from base models (n_samples, n_features)
            y: True labels (n_samples,)
            
        Returns:
            Dictionary with training metrics
        """
        raise NotImplementedError
    
    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X_meta: Meta-features
            
        Returns:
            Predicted labels
        """
        raise NotImplementedError
    
    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X_meta: Meta-features
            
        Returns:
            Class probabilities
        """
        raise NotImplementedError
    
    def save(self, save_path: Path) -> None:
        """Save model to disk."""
        raise NotImplementedError
    
    @classmethod
    def load(cls, load_path: Path):
        """Load model from disk."""
        raise NotImplementedError


# ============================================================================
# LOGISTIC REGRESSION META-LEARNER
# ============================================================================

class LogisticMetaLearner(MetaLearner):
    """
    Logistic Regression meta-learner.
    
    Simple and interpretable. Good baseline for stacking.
    """
    
    def __init__(
        self,
        C: float = META_LEARNER_CONFIG['C'],
        penalty: str = META_LEARNER_CONFIG['penalty'],
        max_iter: int = META_LEARNER_CONFIG['max_iter'],
        random_state: int = RANDOM_SEED
    ):
        """
        Initialize Logistic Regression meta-learner.
        
        Args:
            C: Inverse regularization strength
            penalty: Regularization penalty ('l1', 'l2', 'elasticnet', 'none')
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.C = C
        self.penalty = penalty
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            max_iter=max_iter,
            random_state=random_state,
            solver='lbfgs' if penalty == 'l2' else 'saga',
            n_jobs=-1
        )
        
        logger.info(f"Initialized Logistic Meta-Learner (C={C}, penalty={penalty})")
    
    def fit(
        self,
        X_meta: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Fit logistic regression on meta-features.
        
        Args:
            X_meta: Meta-features (n_samples, n_features)
            y: Labels (n_samples,)
            X_val: Validation meta-features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Training Logistic Meta-Learner on {len(X_meta)} samples...")
        logger.info(f"Meta-features shape: {X_meta.shape}")
        
        try:
            # Fit model
            self.model.fit(X_meta, y)
            
            # Training accuracy
            train_acc = self.model.score(X_meta, y)
            logger.info(f"Training accuracy: {train_acc:.4f}")
            
            metrics = {'train_accuracy': train_acc}
            
            # Validation accuracy
            if X_val is not None and y_val is not None:
                val_acc = self.model.score(X_val, y_val)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                metrics['val_accuracy'] = val_acc
            
            # Log feature coefficients
            if hasattr(self.model, 'coef_'):
                logger.info(f"Feature coefficients shape: {self.model.coef_.shape}")
                logger.info(f"Coefficient norms: {np.linalg.norm(self.model.coef_, axis=1)}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error training meta-learner: {e}")
            raise
    
    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X_meta)
    
    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X_meta)
    
    def get_feature_importance(self) -> np.ndarray:
        """
        Get feature importance (coefficient magnitudes).
        
        Returns:
            Feature importance scores
        """
        if hasattr(self.model, 'coef_'):
            return np.abs(self.model.coef_[0])
        else:
            raise ValueError("Model has no coefficients")
    
    def save(self, save_path: Path) -> None:
        """Save model to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': {
                'C': self.C,
                'penalty': self.penalty,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
            }
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Meta-learner saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'LogisticMetaLearner':
        """Load model from disk."""
        logger.info(f"Loading meta-learner from: {load_path}")
        
        model_data = joblib.load(load_path)
        config = model_data['config']
        
        # Create instance
        meta_learner = cls(
            C=config['C'],
            penalty=config['penalty'],
            max_iter=config['max_iter'],
            random_state=config['random_state']
        )
        
        # Restore trained model
        meta_learner.model = model_data['model']
        
        logger.info("Meta-learner loaded successfully")
        return meta_learner


# ============================================================================
# XGBOOST META-LEARNER
# ============================================================================

class XGBoostMetaLearner(MetaLearner):
    """
    XGBoost meta-learner.
    
    Gradient boosting is excellent for stacking as it can:
    - Handle complex interactions between base models
    - Provide feature importance
    - Resist overfitting with proper regularization
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.0,
        reg_lambda: float = 1.0,
        random_state: int = RANDOM_SEED,
        use_gpu: bool = False
    ):
        """
        Initialize XGBoost meta-learner.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            reg_alpha: L1 regularization
            reg_lambda: L2 regularization
            random_state: Random seed
            use_gpu: Whether to use GPU acceleration
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.use_gpu = use_gpu
        
        # Determine tree method based on device
        if use_gpu:
            tree_method = 'gpu_hist'
            predictor = 'gpu_predictor'
        else:
            tree_method = 'hist'
            predictor = 'cpu_predictor'
        
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            tree_method=tree_method,
            predictor=predictor,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        logger.info(f"Initialized XGBoost Meta-Learner")
        logger.info(f"  n_estimators={n_estimators}, max_depth={max_depth}")
        logger.info(f"  learning_rate={learning_rate}")
        logger.info(f"  Device: {'GPU' if use_gpu else 'CPU'}")
    
    def fit(
        self,
        X_meta: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        early_stopping_rounds: int = 10,
        verbose: bool = False
    ) -> Dict[str, float]:
        """
        Fit XGBoost on meta-features.
        
        Args:
            X_meta: Meta-features (n_samples, n_features)
            y: Labels (n_samples,)
            X_val: Validation meta-features (optional)
            y_val: Validation labels (optional)
            early_stopping_rounds: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Training XGBoost Meta-Learner on {len(X_meta)} samples...")
        logger.info(f"Meta-features shape: {X_meta.shape}")
        
        try:
            # Prepare evaluation set
            eval_set = []
            if X_val is not None and y_val is not None:
                eval_set = [(X_meta, y), (X_val, y_val)]
            else:
                eval_set = [(X_meta, y)]
            
            # Fit model
            self.model.fit(
                X_meta,
                y,
                eval_set=eval_set,
                early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
                verbose=verbose
            )
            
            # Training accuracy
            train_acc = self.model.score(X_meta, y)
            logger.info(f"Training accuracy: {train_acc:.4f}")
            logger.info(f"Best iteration: {self.model.best_iteration}")
            
            metrics = {
                'train_accuracy': train_acc,
                'best_iteration': self.model.best_iteration
            }
            
            # Validation accuracy
            if X_val is not None and y_val is not None:
                val_acc = self.model.score(X_val, y_val)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                metrics['val_accuracy'] = val_acc
            
            # Log feature importance
            feature_importance = self.get_feature_importance()
            logger.info(f"Feature importance: {feature_importance}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error training XGBoost meta-learner: {e}")
            raise
    
    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X_meta)
    
    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X_meta)
    
    def get_feature_importance(self, importance_type: str = 'weight') -> np.ndarray:
        """
        Get feature importance from XGBoost.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            Feature importance scores
        """
        return self.model.feature_importances_
    
    def save(self, save_path: Path) -> None:
        """Save model to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(str(save_path.with_suffix('.json')))
        
        # Save config separately
        config_data = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'use_gpu': self.use_gpu,
        }
        
        joblib.dump(config_data, save_path.with_suffix('.config.pkl'))
        logger.info(f"XGBoost meta-learner saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'XGBoostMetaLearner':
        """Load model from disk."""
        logger.info(f"Loading XGBoost meta-learner from: {load_path}")
        
        # Load config
        config = joblib.load(load_path.with_suffix('.config.pkl'))
        
        # Create instance
        meta_learner = cls(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            subsample=config['subsample'],
            colsample_bytree=config['colsample_bytree'],
            reg_alpha=config['reg_alpha'],
            reg_lambda=config['reg_lambda'],
            random_state=config['random_state'],
            use_gpu=config['use_gpu']
        )
        
        # Load XGBoost model
        meta_learner.model.load_model(str(load_path.with_suffix('.json')))
        
        logger.info("XGBoost meta-learner loaded successfully")
        return meta_learner


# ============================================================================
# MLP META-LEARNER
# ============================================================================

class MLPMetaLearner(MetaLearner):
    """
    Multi-layer Perceptron meta-learner.
    
    More flexible than logistic regression, can capture
    non-linear interactions between base models.
    """
    
    def __init__(
        self,
        hidden_layer_sizes: tuple = META_LEARNER_CONFIG['hidden_layer_sizes'],
        activation: str = META_LEARNER_CONFIG['activation'],
        solver: str = META_LEARNER_CONFIG['solver'],
        alpha: float = META_LEARNER_CONFIG['alpha'],
        max_iter: int = META_LEARNER_CONFIG['max_iter'],
        random_state: int = RANDOM_SEED
    ):
        """
        Initialize MLP meta-learner.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            activation: Activation function ('relu', 'tanh', 'logistic')
            solver: Optimizer ('adam', 'sgd', 'lbfgs')
            alpha: L2 regularization parameter
            max_iter: Maximum iterations
            random_state: Random seed
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        logger.info(f"Initialized MLP Meta-Learner")
        logger.info(f"  Hidden layers: {hidden_layer_sizes}")
        logger.info(f"  Activation: {activation}")
    
    def fit(
        self,
        X_meta: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Fit MLP on meta-features.
        
        Args:
            X_meta: Meta-features
            y: Labels
            X_val: Validation meta-features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with metrics
        """
        logger.info(f"Training MLP Meta-Learner on {len(X_meta)} samples...")
        logger.info(f"Meta-features shape: {X_meta.shape}")
        
        try:
            # Fit model
            self.model.fit(X_meta, y)
            
            # Training accuracy
            train_acc = self.model.score(X_meta, y)
            logger.info(f"Training accuracy: {train_acc:.4f}")
            logger.info(f"Training iterations: {self.model.n_iter_}")
            
            metrics = {'train_accuracy': train_acc}
            
            # Validation accuracy
            if X_val is not None and y_val is not None:
                val_acc = self.model.score(X_val, y_val)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                metrics['val_accuracy'] = val_acc
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error training MLP meta-learner: {e}")
            raise
    
    def predict(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict labels."""
        return self.model.predict(X_meta)
    
    def predict_proba(self, X_meta: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X_meta)
    
    def save(self, save_path: Path) -> None:
        """Save model to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'config': {
                'hidden_layer_sizes': self.hidden_layer_sizes,
                'activation': self.activation,
                'solver': self.solver,
                'alpha': self.alpha,
                'max_iter': self.max_iter,
                'random_state': self.random_state,
            }
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"MLP meta-learner saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'MLPMetaLearner':
        """Load model from disk."""
        logger.info(f"Loading MLP meta-learner from: {load_path}")
        
        model_data = joblib.load(load_path)
        config = model_data['config']
        
        # Create instance
        meta_learner = cls(
            hidden_layer_sizes=config['hidden_layer_sizes'],
            activation=config['activation'],
            solver=config['solver'],
            alpha=config['alpha'],
            max_iter=config['max_iter'],
            random_state=config['random_state']
        )
        
        # Restore trained model
        meta_learner.model = model_data['model']
        
        logger.info("MLP meta-learner loaded successfully")
        return meta_learner


# ============================================================================
# PYTORCH MLP META-LEARNER (Alternative)
# ============================================================================

class PyTorchMLPMetaLearner(nn.Module):
    """
    PyTorch-based MLP meta-learner.
    
    Alternative to sklearn MLP with more control over architecture.
    Useful for GPU acceleration with large meta-feature sets.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [64, 32],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize PyTorch MLP meta-learner.
        
        Args:
            input_dim: Number of meta-features
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"Initialized PyTorch MLP Meta-Learner")
        logger.info(f"  Architecture: {input_dim} -> {hidden_dims} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Meta-features (batch_size, input_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Predict class probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def save(self, save_path: Path) -> None:
        """Save model to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'num_classes': self.num_classes,
                'dropout': self.dropout_rate,
            }
        }, save_path)
        
        logger.info(f"PyTorch meta-learner saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path, device: torch.device = DEVICE) -> 'PyTorchMLPMetaLearner':
        """Load model from disk."""
        logger.info(f"Loading PyTorch meta-learner from: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['config']
        
        # Create model
        model = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info("PyTorch meta-learner loaded successfully")
        return model


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_meta_learner(
    model_type: str = 'logistic',
    **kwargs
) -> MetaLearner:
    """
    Factory function to create meta-learner.
    
    Args:
        model_type: Type of meta-learner ('logistic', 'mlp', 'xgboost')
        **kwargs: Additional arguments for meta-learner
        
    Returns:
        Meta-learner instance
        
    Raises:
        ValueError: If model_type is invalid
    """
    model_type = model_type.lower()
    
    if model_type == 'logistic':
        return LogisticMetaLearner(**kwargs)
    elif model_type == 'mlp':
        return MLPMetaLearner(**kwargs)
    elif model_type == 'xgboost':
        return XGBoostMetaLearner(**kwargs)
    else:
        raise ValueError(f"Invalid meta-learner type: {model_type}. Choose from: logistic, mlp, xgboost")


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing meta-learners."""
    logger.info("=" * 80)
    logger.info("TESTING META-LEARNERS")
    logger.info("=" * 80)
    
    # Create synthetic meta-features
    np.random.seed(42)
    n_samples = 100
    n_features = 8  # 4 models x 2 classes = 8 probability features
    
    X_meta = np.random.rand(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)
    
    X_val = np.random.rand(20, n_features)
    y_val = np.random.randint(0, 2, 20)
    
    logger.info(f"Synthetic data: {X_meta.shape}")
    
    # Test Logistic Meta-Learner
    logger.info("\n1. Testing Logistic Meta-Learner...")
    logistic = LogisticMetaLearner(C=1.0, penalty='l2')
    metrics = logistic.fit(X_meta, y, X_val, y_val)
    logger.info(f"Metrics: {metrics}")
    
    y_pred = logistic.predict(X_val)
    y_prob = logistic.predict_proba(X_val)
    logger.info(f"Predictions shape: {y_pred.shape}")
    logger.info(f"Probabilities shape: {y_prob.shape}")
    
    # Test XGBoost Meta-Learner
    logger.info("\n2. Testing XGBoost Meta-Learner...")
    xgboost = XGBoostMetaLearner(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        use_gpu=False
    )
    metrics = xgboost.fit(X_meta, y, X_val, y_val, early_stopping_rounds=10)
    logger.info(f"Metrics: {metrics}")
    
    y_pred_xgb = xgboost.predict(X_val)
    y_prob_xgb = xgboost.predict_proba(X_val)
    logger.info(f"XGBoost predictions shape: {y_pred_xgb.shape}")
    logger.info(f"XGBoost probabilities shape: {y_prob_xgb.shape}")
    logger.info(f"XGBoost feature importance: {xgboost.get_feature_importance()}")
    
    # Test save/load
    save_path = SAVED_MODELS_DIR / "test_meta_learner_xgboost"
    xgboost.save(save_path)
    xgboost_loaded = XGBoostMetaLearner.load(save_path)
    logger.info("XGBoost save/load test passed")
    
    # Test MLP Meta-Learner
    logger.info("\n3. Testing MLP Meta-Learner...")
    mlp = MLPMetaLearner(hidden_layer_sizes=(32, 16))
    metrics = mlp.fit(X_meta, y, X_val, y_val)
    logger.info(f"Metrics: {metrics}")
    
    # Test PyTorch MLP
    logger.info("\n4. Testing PyTorch MLP Meta-Learner...")
    pytorch_mlp = PyTorchMLPMetaLearner(
        input_dim=n_features,
        hidden_dims=[32, 16],
        num_classes=2
    )
    
    X_tensor = torch.tensor(X_meta[:10], dtype=torch.float32)
    pytorch_mlp.eval()
    with torch.no_grad():
        logits = pytorch_mlp(X_tensor)
        probs = pytorch_mlp.predict_proba(X_tensor)
    
    logger.info(f"PyTorch logits shape: {logits.shape}")
    logger.info(f"PyTorch probabilities shape: {probs.shape}")
    
    # Test factory function
    logger.info("\n5. Testing factory function...")
    meta_log = create_meta_learner('logistic', C=1.0)
    meta_xgb = create_meta_learner('xgboost', n_estimators=50)
    meta_mlp = create_meta_learner('mlp', hidden_layer_sizes=(32, 16))
    logger.info("Factory function test passed")
    
    logger.info("\n" + "=" * 80)
    logger.info("META-LEARNER TESTS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

