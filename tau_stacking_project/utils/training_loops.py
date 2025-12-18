"""
Training loops and utilities for PyTorch models.
"""

import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.
    
    Usage:
        early_stopping = EarlyStopping(patience=7)
        for epoch in range(num_epochs):
            val_loss = train_one_epoch(...)
            if early_stopping(model, val_loss):
                break
    """
    def __init__(self, patience=7, min_delta=0.0, restore_best_weights=True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best model weights
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_model = None
        self.best_loss = None
        self.counter = 0
        self.status = ""
        
    def __call__(self, model, val_loss):
        """
        Check if training should stop.
        
        Args:
            model: PyTorch model
            val_loss: Current validation loss
            
        Returns:
            bool: True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
            
        if self.best_loss - val_loss > self.min_delta:
            # Improvement
            self.best_loss = val_loss
            self.counter = 0
            self.best_model = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.status = f"Improved"
        else:
            # No improvement
            self.counter += 1
            self.status = f"{self.counter}/{self.patience}"
            
            if self.counter >= self.patience:
                # Stop training
                if self.restore_best_weights:
                    model.load_state_dict(self.best_model)
                return True
                
        return False


def train_torch_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    num_epochs,
    device=None,
    early_stopping=None,
    scheduler=None,
    verbose=True
):
    """
    Generic training loop for PyTorch models.
    
    Args:
        model: PyTorch model
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of training epochs
        device: Device to train on (default: auto-detect)
        early_stopping: EarlyStopping instance (optional)
        scheduler: Learning rate scheduler (optional)
        verbose: Whether to print progress
    
    Returns:
        dict: Training history with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = model.to(device)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # ============================================
        # TRAINING PHASE
        # ============================================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        if verbose:
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]', leave=False)
        else:
            train_pbar = train_loader
        
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if verbose and hasattr(train_pbar, 'set_postfix'):
                train_pbar.set_postfix({
                    'loss': train_loss/(batch_idx+1), 
                    'acc': 100.*train_correct/train_total
                })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # ============================================
        # VALIDATION PHASE
        # ============================================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}: '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # Early stopping
        if early_stopping is not None:
            if early_stopping(model, avg_val_loss):
                if verbose:
                    print(f'✋ Early stopping triggered at epoch {epoch+1}')
                break
    
    return history


def evaluate_model(model, test_loader, device=None):
    """
    Evaluate model on test set.
    
    Args:
        model: PyTorch model
        test_loader: Test DataLoader
        device: Device to evaluate on
    
    Returns:
        tuple: (accuracy, predictions, true_labels, probabilities)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100. * correct / total
    
    return accuracy, np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def train_sklearn_model(model, X_train, y_train, X_val=None, y_val=None, verbose=True):
    """
    Training wrapper for scikit-learn models.
    
    Args:
        model: Sklearn model with fit() method
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        verbose: Whether to print progress
    
    Returns:
        Trained model
    """
    if verbose:
        print("Training sklearn model...")
    
    model.fit(X_train, y_train)
    
    if verbose:
        train_score = model.score(X_train, y_train)
        print(f"✅ Training completed. Train accuracy: {train_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = model.score(X_val, y_val)
            print(f"   Validation accuracy: {val_score:.4f}")
    
    return model

def predict_with_torch_model(model, data_loader, device=None):
    """
    Generate predictions using a PyTorch model.
    
    Args:
        model: PyTorch model
        data_loader: DataLoader with data to predict
        device: Device to use for prediction
    
    Returns:
        tuple: (predictions, probabilities)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 
                            'mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for inputs in tqdm(data_loader, desc='Predicting'):
            # Handle both (inputs,) and (inputs, labels) formats
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
            
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            # Get probabilities
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_probabilities)

