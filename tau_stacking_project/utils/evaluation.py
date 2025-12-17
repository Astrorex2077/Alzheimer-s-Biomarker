# Metrics and evaluation helpers
"""
Model evaluation and metrics utilities.

This module provides:
- Classification metrics computation
- ROC and PR curve generation
- Confusion matrix plotting
- Model comparison utilities
- Results export functions

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    average_precision_score,
)
import json

from utils.config import (
    METRICS_DIR,
    PREDICTIONS_DIR,
    EVALUATION_CONFIG,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = EVALUATION_CONFIG['plot_dpi']


# ============================================================================
# CLASSIFICATION METRICS
# ============================================================================

def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = EVALUATION_CONFIG['threshold'],
    average: str = 'binary'
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (for ROC-AUC)
        threshold: Classification threshold
        average: Averaging method for multi-class
        
    Returns:
        Dictionary of metric name -> value
    """
    logger.info("Computing classification metrics...")
    
    metrics = {}
    
    try:
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, average=average, zero_division=0)
        
        # ROC-AUC (requires probabilities)
        if y_prob is not None:
            try:
                if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                    # Multi-class: use probability of positive class
                    y_prob_pos = y_prob[:, 1]
                else:
                    y_prob_pos = y_prob.ravel()
                
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob_pos)
                metrics['average_precision'] = average_precision_score(y_true, y_prob_pos)
            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics['roc_auc'] = None
                metrics['average_precision'] = None
        else:
            metrics['roc_auc'] = None
            metrics['average_precision'] = None
        
        # Confusion matrix elements
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Same as recall
        
        # False rates
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        # Log results
        logger.info("Metrics computed:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics['roc_auc']:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise


def compute_metrics_by_class(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Compute per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        
    Returns:
        DataFrame with per-class metrics
    """
    logger.info("Computing per-class metrics...")
    
    # Get classification report as dict
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Convert to DataFrame
    df_metrics = pd.DataFrame(report).T
    
    logger.info(f"Per-class metrics:\n{df_metrics}")
    
    return df_metrics


# ============================================================================
# ROC CURVE
# ============================================================================

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Path] = None,
    show: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plot ROC curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Tuple of (fpr, tpr, auc_score)
    """
    logger.info("Plotting ROC curve...")
    
    try:
        # Handle multi-class probabilities
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.ravel()
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
            logger.info(f"ROC curve saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return fpr, tpr, auc_score
    
    except Exception as e:
        logger.error(f"Error plotting ROC curve: {e}")
        raise


def plot_multiple_roc_curves(
    y_true: np.ndarray,
    y_probs_dict: Dict[str, np.ndarray],
    title: str = "ROC Curve Comparison",
    save_path: Optional[Path] = None,
    show: bool = False
) -> Dict[str, float]:
    """
    Plot multiple ROC curves for model comparison.
    
    Args:
        y_true: True labels
        y_probs_dict: Dictionary of model_name -> probabilities
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Dictionary of model_name -> AUC score
    """
    logger.info(f"Plotting ROC curves for {len(y_probs_dict)} models...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    auc_scores = {}
    colors = plt.cm.Set2(np.linspace(0, 1, len(y_probs_dict)))
    
    for (model_name, y_prob), color in zip(y_probs_dict.items(), colors):
        # Handle multi-class probabilities
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.ravel()
        
        # Compute ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_score = roc_auc_score(y_true, y_prob)
        auc_scores[model_name] = auc_score
        
        # Plot
        ax.plot(fpr, tpr, linewidth=2, color=color, 
                label=f'{model_name} (AUC = {auc_score:.4f})')
    
    # Random baseline
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
        logger.info(f"ROC comparison saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return auc_scores


# ============================================================================
# PRECISION-RECALL CURVE
# ============================================================================

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Optional[Path] = None,
    show: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Plot Precision-Recall curve.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Tuple of (precision, recall, average_precision)
    """
    logger.info("Plotting Precision-Recall curve...")
    
    try:
        # Handle multi-class probabilities
        if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
            y_prob = y_prob[:, 1]
        else:
            y_prob = y_prob.ravel()
        
        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_precision = average_precision_score(y_true, y_prob)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.plot(recall, precision, linewidth=2, 
                label=f'PR (AP = {avg_precision:.4f})')
        
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
            logger.info(f"PR curve saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return precision, recall, avg_precision
    
    except Exception as e:
        logger.error(f"Error plotting PR curve: {e}")
        raise


# ============================================================================
# CONFUSION MATRIX
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = False,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    show: bool = False
) -> np.ndarray:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names of classes
        normalize: Whether to normalize
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Confusion matrix array
    """
    logger.info("Plotting confusion matrix...")
    
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        else:
            fmt = 'd'
        
        # Class names
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(cm))]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
            logger.info(f"Confusion matrix saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return cm
    
    except Exception as e:
        logger.error(f"Error plotting confusion matrix: {e}")
        raise


# ============================================================================
# MODEL COMPARISON
# ============================================================================

def plot_accuracy_bar(
    metrics_dict: Dict[str, float],
    title: str = "Model Accuracy Comparison",
    save_path: Optional[Path] = None,
    show: bool = False
) -> None:
    """
    Plot bar chart comparing model accuracies.
    
    Args:
        metrics_dict: Dictionary of model_name -> accuracy
        title: Plot title
        save_path: Path to save figure
        show: Whether to display plot
    """
    logger.info(f"Plotting accuracy comparison for {len(metrics_dict)} models...")
    
    try:
        # Sort by accuracy
        sorted_items = sorted(metrics_dict.items(), key=lambda x: x[1], reverse=True)
        models, accuracies = zip(*sorted_items)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
        bars = ax.barh(models, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            width = bar.get_width()
            ax.text(width, bar.get_y() + bar.get_height()/2,
                   f'{acc:.2%}',
                   ha='left', va='center', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=EVALUATION_CONFIG['plot_dpi'], bbox_inches='tight')
            logger.info(f"Accuracy bar chart saved: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    except Exception as e:
        logger.error(f"Error plotting accuracy bar chart: {e}")
        raise


def create_metrics_comparison_table(
    metrics_dict: Dict[str, Dict[str, float]],
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create comparison table of metrics across models.
    
    Args:
        metrics_dict: Dictionary of model_name -> metrics_dict
        output_path: Path to save CSV
        
    Returns:
        DataFrame with comparison
    """
    logger.info(f"Creating metrics comparison for {len(metrics_dict)} models...")
    
    try:
        # Create DataFrame
        df = pd.DataFrame(metrics_dict).T
        
        # Round values
        df = df.round(4)
        
        # Sort by a key metric (e.g., accuracy)
        if 'accuracy' in df.columns:
            df = df.sort_values('accuracy', ascending=False)
        
        logger.info(f"Metrics comparison:\n{df}")
        
        # Save to CSV
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path)
            logger.info(f"Metrics comparison saved: {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error creating metrics comparison: {e}")
        raise


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_predictions(
    protein_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    output_path: Path = PREDICTIONS_DIR / "predictions.csv"
) -> pd.DataFrame:
    """
    Export predictions to CSV.
    
    Args:
        protein_ids: List of protein IDs
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities
        output_path: Output CSV path
        
    Returns:
        DataFrame with predictions
    """
    logger.info(f"Exporting predictions for {len(protein_ids)} proteins...")
    
    try:
        data = {
            'protein_id': protein_ids,
            'true_label': y_true,
            'predicted_label': y_pred,
        }
        
        if y_prob is not None:
            if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                data['probability_class_0'] = y_prob[:, 0]
                data['probability_class_1'] = y_prob[:, 1]
            else:
                data['probability'] = y_prob.ravel()
        
        df = pd.DataFrame(data)
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Predictions saved: {output_path}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error exporting predictions: {e}")
        raise


def save_metrics_json(
    metrics: Dict[str, Union[float, int]],
    output_path: Path = METRICS_DIR / "metrics.json"
) -> None:
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        output_path: Output JSON path
    """
    logger.info("Saving metrics to JSON...")
    
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved: {output_path}")
    
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing evaluation utilities."""
    logger.info("=" * 80)
    logger.info("TESTING EVALUATION UTILITIES")
    logger.info("=" * 80)
    
    # Create synthetic data
    np.random.seed(42)
    n_samples = 200
    y_true = np.random.randint(0, 2, n_samples)
    y_prob = np.random.rand(n_samples, 2)
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)
    y_pred = np.argmax(y_prob, axis=1)
    
    logger.info(f"Created synthetic data: {n_samples} samples")
    
    # Test metrics computation
    logger.info("\n1. Testing metrics computation...")
    metrics = compute_classification_metrics(y_true, y_pred, y_prob)
    
    # Test plots (without showing)
    logger.info("\n2. Testing plot generation...")
    plot_roc_curve(y_true, y_prob, show=False)
    plot_precision_recall_curve(y_true, y_prob, show=False)
    plot_confusion_matrix(y_true, y_pred, class_names=['Stable', 'Misfolding'], show=False)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION UTILITIES TEST COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

