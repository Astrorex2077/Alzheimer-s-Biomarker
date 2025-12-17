# Model A: ProtBERT frozen + SVM / dense head
"""
Model A: ProtBERT Frozen + SVM/Dense Classifier

This model uses frozen ProtBERT embeddings with a traditional
classifier (SVM or small dense network) on top.

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

from utils.config import (
    PROTBERT_CONFIG,
    SVM_CONFIG,
    DEVICE,
    SAVED_MODELS_DIR,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROTBERT FROZEN + SVM
# ============================================================================

class ProtBERTFrozenSVM:
    """
    ProtBERT embeddings (frozen) + SVM classifier.
    
    This model uses pre-computed ProtBERT embeddings and trains
    an SVM classifier on top. No gradient updates to ProtBERT.
    """
    
    def __init__(
        self,
        kernel: str = SVM_CONFIG['kernel'],
        C: float = SVM_CONFIG['C'],
        gamma: str = SVM_CONFIG['gamma'],
        probability: bool = True,
        random_state: int = SVM_CONFIG['random_state'],
        normalize: bool = True
    ):
        """
        Initialize ProtBERT + SVM model.
        
        Args:
            kernel: SVM kernel ('rbf', 'linear', 'poly', 'sigmoid')
            C: Regularization parameter
            gamma: Kernel coefficient
            probability: Enable probability estimates
            random_state: Random seed
            normalize: Whether to normalize embeddings
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.probability = probability
        self.random_state = random_state
        self.normalize = normalize
        
        # Initialize SVM
        self.svm = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=probability,
            random_state=random_state,
            cache_size=SVM_CONFIG.get('cache_size', 1000)
        )
        
        # Scaler for normalization
        self.scaler = StandardScaler() if normalize else None
        
        logger.info(f"Initialized ProtBERT+SVM with kernel={kernel}, C={C}")
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Fit SVM on ProtBERT embeddings.
        
        Args:
            X_train: Training embeddings (n_samples, embedding_dim)
            y_train: Training labels
            X_val: Validation embeddings (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary with training metrics
        """
        logger.info(f"Training SVM on {len(X_train)} samples...")
        logger.info(f"Embedding dimension: {X_train.shape[1]}")
        
        try:
            # Normalize embeddings
            if self.normalize:
                logger.info("Normalizing embeddings...")
                X_train_scaled = self.scaler.fit_transform(X_train)
            else:
                X_train_scaled = X_train
            
            # Train SVM
            logger.info("Fitting SVM...")
            self.svm.fit(X_train_scaled, y_train)
            
            # Evaluate on training set
            train_acc = self.svm.score(X_train_scaled, y_train)
            logger.info(f"Training accuracy: {train_acc:.4f}")
            
            metrics = {'train_accuracy': train_acc}
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                if self.normalize:
                    X_val_scaled = self.scaler.transform(X_val)
                else:
                    X_val_scaled = X_val
                
                val_acc = self.svm.score(X_val_scaled, y_val)
                logger.info(f"Validation accuracy: {val_acc:.4f}")
                metrics['val_accuracy'] = val_acc
            
            logger.info("SVM training completed successfully")
            return metrics
        
        except Exception as e:
            logger.error(f"Error during SVM training: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
            
        Returns:
            Predicted labels
        """
        if self.normalize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.svm.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Embeddings (n_samples, embedding_dim)
            
        Returns:
            Class probabilities (n_samples, n_classes)
        """
        if not self.probability:
            raise ValueError("Model was not trained with probability=True")
        
        if self.normalize:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        return self.svm.predict_proba(X_scaled)
    
    def save(self, save_path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'svm': self.svm,
            'scaler': self.scaler,
            'config': {
                'kernel': self.kernel,
                'C': self.C,
                'gamma': self.gamma,
                'normalize': self.normalize,
            }
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path) -> 'ProtBERTFrozenSVM':
        """
        Load model from disk.
        
        Args:
            load_path: Path to load model from
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from: {load_path}")
        
        model_data = joblib.load(load_path)
        
        # Create new instance
        model = cls(
            kernel=model_data['config']['kernel'],
            C=model_data['config']['C'],
            gamma=model_data['config']['gamma'],
            normalize=model_data['config']['normalize'],
        )
        
        # Restore trained components
        model.svm = model_data['svm']
        model.scaler = model_data['scaler']
        
        logger.info("Model loaded successfully")
        return model


# ============================================================================
# PROTBERT FROZEN + DENSE NETWORK
# ============================================================================

class ProtBERTFrozenDense(nn.Module):
    """
    ProtBERT embeddings (frozen) + small dense classifier.
    
    Alternative to SVM - uses a small neural network on top
    of frozen ProtBERT embeddings.
    """
    
    def __init__(
        self,
        input_dim: int = PROTBERT_CONFIG['embedding_dim'],
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize dense classifier.
        
        Args:
            input_dim: ProtBERT embedding dimension
            hidden_dims: List of hidden layer dimensions
            num_classes: Number of output classes
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Build network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
        
        logger.info(f"Initialized ProtBERT+Dense: {input_dim} -> {hidden_dims} -> {num_classes}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: ProtBERT embeddings (batch_size, embedding_dim)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.classifier(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: ProtBERT embeddings
            
        Returns:
            Probabilities (batch_size, num_classes)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)
    
    def save(self, save_path: Path) -> None:
        """
        Save model to disk.
        
        Args:
            save_path: Path to save model
        """
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.input_dim,
                'hidden_dims': self.hidden_dims,
                'num_classes': self.num_classes,
                'dropout': self.dropout,
            }
        }, save_path)
        
        logger.info(f"Model saved: {save_path}")
    
    @classmethod
    def load(cls, load_path: Path, device: torch.device = DEVICE) -> 'ProtBERTFrozenDense':
        """
        Load model from disk.
        
        Args:
            load_path: Path to load model from
            device: Device to load model to
            
        Returns:
            Loaded model instance
        """
        logger.info(f"Loading model from: {load_path}")
        
        checkpoint = torch.load(load_path, map_location=device)
        config = checkpoint['config']
        
        # Create model instance
        model = cls(
            input_dim=config['input_dim'],
            hidden_dims=config['hidden_dims'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        
        logger.info("Model loaded successfully")
        return model


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing Model A."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL A: PROTBERT FROZEN + SVM")
    logger.info("=" * 80)
    
    # Create synthetic embeddings
    np.random.seed(42)
    n_train = 100
    n_val = 20
    embedding_dim = 1024
    
    X_train = np.random.randn(n_train, embedding_dim)
    y_train = np.random.randint(0, 2, n_train)
    X_val = np.random.randn(n_val, embedding_dim)
    y_val = np.random.randint(0, 2, n_val)
    
    logger.info(f"Created synthetic data:")
    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val: {X_val.shape}")
    
    # Test SVM model
    logger.info("\n1. Testing ProtBERT+SVM...")
    model_svm = ProtBERTFrozenSVM(kernel='rbf', normalize=True)
    metrics = model_svm.fit(X_train, y_train, X_val, y_val)
    logger.info(f"Metrics: {metrics}")
    
    # Test predictions
    y_pred = model_svm.predict(X_val)
    y_prob = model_svm.predict_proba(X_val)
    logger.info(f"Predictions shape: {y_pred.shape}")
    logger.info(f"Probabilities shape: {y_prob.shape}")
    
    # Test save/load
    save_path = SAVED_MODELS_DIR / "test_protbert_svm.pkl"
    model_svm.save(save_path)
    model_loaded = ProtBERTFrozenSVM.load(save_path)
    logger.info("Save/load test passed")
    
    # Test Dense model
    logger.info("\n2. Testing ProtBERT+Dense...")
    model_dense = ProtBERTFrozenDense(
        input_dim=embedding_dim,
        hidden_dims=[256, 128],
        num_classes=2
    )
    logger.info(f"Model: {model_dense}")
    
    # Test forward pass
    X_tensor = torch.tensor(X_train[:10], dtype=torch.float32)
    outputs = model_dense(X_tensor)
    logger.info(f"Output shape: {outputs.shape}")
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL A TESTS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

