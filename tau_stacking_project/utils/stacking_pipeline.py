# High-level stacking pipeline
"""
Generic training loops for PyTorch and scikit-learn models.

This module provides reusable training functions with:
- Early stopping
- Checkpointing
- Progress tracking
- Metrics logging
- Learning rate scheduling

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.base import BaseEstimator
from tqdm import tqdm
import json

from utils.config import (
    SAVED_MODELS_DIR,
    LOGS_DIR,
    TRAINING_CONFIG,
    DEVICE,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# EARLY STOPPING
# ============================================================================

class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = 'min',
        verbose: bool = True
    ):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
        if mode == 'min':
            self.is_better = lambda new, best: new < best - min_delta
        else:  # 'max'
            self.is_better = lambda new, best: new > best + min_delta
    
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current validation score
            epoch: Current epoch number
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.is_better(score, self.best_score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                logger.info(f"Validation improved to {score:.4f}")
        else:
            self.counter += 1
            if self.verbose:
                logger.info(f"No improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                return True
        
        return False


# ============================================================================
# PYTORCH TRAINING LOOP
# ============================================================================

def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    num_epochs: int,
    device: torch.device = DEVICE,
    save_path: Optional[Path] = None,
    early_stopping_patience: int = TRAINING_CONFIG['early_stopping_patience'],
    scheduler: Optional[Any] = None,
    gradient_clip: Optional[float] = TRAINING_CONFIG['gradient_clip_value'],
    log_interval: int = 10,
) -> Dict[str, List[float]]:
    """
    Generic PyTorch training loop with early stopping and checkpointing.
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        device: Device to use
        save_path: Path to save best model
        early_stopping_patience: Patience for early stopping
        scheduler: Learning rate scheduler
        gradient_clip: Gradient clipping value (None to disable)
        log_interval: Log every N batches
        
    Returns:
        Dictionary with training history
    """
    logger.info("=" * 80)
    logger.info("STARTING PYTORCH TRAINING")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Val batches: {len(val_loader)}")
    
    # Move model to device
    model = model.to(device)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=early_stopping_patience,
        mode='min',
        verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'learning_rates': [],
    }
    
    best_val_loss = float('inf')
    
    try:
        for epoch in range(num_epochs):
            logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
            logger.info("-" * 40)
            
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc="Training")
            for batch_idx, batch in enumerate(train_pbar):
                # Get data
                if isinstance(batch, dict):
                    inputs = batch['input'].to(device)
                    labels = batch['label'].to(device)
                else:
                    inputs, labels = batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                
                # Zero gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                # Calculate loss
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if gradient_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                
                # Update weights
                optimizer.step()
                
                # Statistics
                train_loss += loss.item()
                
                # Calculate accuracy (for classification)
                if outputs.shape[1] > 1:  # Multi-class
                    _, predicted = torch.max(outputs, 1)
                else:  # Binary
                    predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Update progress bar
                if (batch_idx + 1) % log_interval == 0:
                    train_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100 * train_correct / train_total:.2f}%'
                    })
            
            # Calculate epoch training metrics
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = 100 * train_correct / train_total
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc="Validation")
                for batch in val_pbar:
                    # Get data
                    if isinstance(batch, dict):
                        inputs = batch['input'].to(device)
                        labels = batch['label'].to(device)
                    else:
                        inputs, labels = batch
                        inputs = inputs.to(device)
                        labels = labels.to(device)
                    
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Handle different output formats
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    
                    # Calculate loss
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Calculate accuracy
                    if outputs.shape[1] > 1:  # Multi-class
                        _, predicted = torch.max(outputs, 1)
                    else:  # Binary
                        predicted = (torch.sigmoid(outputs) > 0.5).long().squeeze()
                    
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            # Calculate epoch validation metrics
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * val_correct / val_total
            
            # Update history
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['train_acc'].append(train_accuracy)
            history['val_acc'].append(val_accuracy)
            
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            history['learning_rates'].append(current_lr)
            
            # Log epoch results
            logger.info(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
            logger.info(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.2f}%")
            logger.info(f"Learning Rate: {current_lr:.6f}")
            
            # Learning rate scheduling
            if scheduler:
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': avg_val_loss,
                        'val_acc': val_accuracy,
                    }, save_path)
                    logger.info(f"Saved best model to {save_path}")
            
            # Early stopping check
            if early_stopping(avg_val_loss, epoch):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        return history
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# ============================================================================
# SKLEARN TRAINING
# ============================================================================

def train_sklearn_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    save_path: Optional[Path] = None,
    model_name: str = "sklearn_model"
) -> Dict[str, float]:
    """
    Train scikit-learn model with validation.
    
    Args:
        model: Scikit-learn model
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        save_path: Path to save model
        model_name: Name for logging
        
    Returns:
        Dictionary with training metrics
    """
    logger.info("=" * 80)
    logger.info(f"TRAINING {model_name.upper()}")
    logger.info("=" * 80)
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    logger.info(f"Features: {X_train.shape[1]}")
    
    try:
        # Train model
        logger.info("Fitting model...")
        model.fit(X_train, y_train)
        
        # Evaluate on training set
        train_score = model.score(X_train, y_train)
        logger.info(f"Training accuracy: {train_score:.4f}")
        
        # Evaluate on validation set
        val_score = model.score(X_val, y_val)
        logger.info(f"Validation accuracy: {val_score:.4f}")
        
        # Save model
        if save_path:
            import joblib
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, save_path)
            logger.info(f"Saved model to {save_path}")
        
        logger.info("=" * 80)
        logger.info(f"{model_name.upper()} TRAINING COMPLETED")
        logger.info("=" * 80)
        
        return {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
        }
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


# ============================================================================
# PREDICTIONS
# ============================================================================

def predict_with_torch_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device = DEVICE,
    return_probs: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate predictions with PyTorch model.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to use
        return_probs: Whether to return probabilities
        
    Returns:
        Predictions (and probabilities if return_probs=True)
    """
    logger.info("Generating predictions...")
    
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Predicting"):
            # Get inputs
            if isinstance(batch, dict):
                inputs = batch['input'].to(device)
            else:
                inputs = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Handle different output formats
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Get probabilities and predictions
            if outputs.shape[1] > 1:  # Multi-class
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
            else:  # Binary
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long().squeeze()
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    predictions = np.concatenate(all_preds)
    
    if return_probs:
        probabilities = np.concatenate(all_probs)
        return predictions, probabilities
    else:
        return predictions


def predict_with_sklearn_model(
    model: BaseEstimator,
    X: np.ndarray,
    return_probs: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Generate predictions with scikit-learn model.
    
    Args:
        model: Scikit-learn model
        X: Features
        return_probs: Whether to return probabilities
        
    Returns:
        Predictions (and probabilities if return_probs=True)
    """
    logger.info(f"Generating predictions for {len(X)} samples...")
    
    predictions = model.predict(X)
    
    if return_probs and hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return predictions, probabilities
    else:
        return predictions


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch: Current epoch
        metrics: Dictionary of metrics
        path: Save path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
    }
    
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    path: Path,
    optimizer: Optional[Optimizer] = None,
    device: torch.device = DEVICE
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model
        path: Checkpoint path
        optimizer: Optimizer (optional)
        device: Device to load to
        
    Returns:
        Checkpoint dictionary
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    logger.info(f"Loading checkpoint: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return checkpoint


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing training utilities."""
    logger.info("=" * 80)
    logger.info("TESTING TRAINING UTILITIES")
    logger.info("=" * 80)
    
    # Test early stopping
    logger.info("\n1. Testing Early Stopping...")
    early_stopping = EarlyStopping(patience=3, mode='min')
    
    val_losses = [0.5, 0.4, 0.45, 0.46, 0.47, 0.48]
    for epoch, loss in enumerate(val_losses):
        should_stop = early_stopping(loss, epoch)
        if should_stop:
            logger.info(f"Would stop at epoch {epoch}")
            break
    
    logger.info("\n" + "=" * 80)
    logger.info("TRAINING UTILITIES TEST COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

