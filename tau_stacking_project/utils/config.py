# Global config and hyperparameters
"""
Central configuration file for tau stacking project.

This module contains all hyperparameters, paths, and constants.
Modify values here rather than hardcoding throughout the project.

Author: Senior ML Engineer
Date: 2025-12-18
"""

from pathlib import Path
from typing import Dict, Any
import torch

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Root directory (automatically detected)
ROOT_DIR = Path(__file__).parent.parent.resolve()

# Data paths
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model paths
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
SAVED_MODELS_DIR = RESULTS_DIR / "models"
EMBEDDINGS_DIR = RESULTS_DIR / "embeddings"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"
LOGS_DIR = RESULTS_DIR / "logs"

# Notebook paths
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# ============================================================================
# DATA FILES
# ============================================================================

FASTA_FILE = RAW_DATA_DIR / "tau_all_species.fasta"
SEQUENCES_CSV = PROCESSED_DATA_DIR / "sequences.csv"
LABELS_CSV = PROCESSED_DATA_DIR / "labels.csv"
SPLITS_CSV = PROCESSED_DATA_DIR / "splits.csv"

# ============================================================================
# MODEL HYPERPARAMETERS
# ============================================================================

# Random seeds for reproducibility
RANDOM_SEED = 42
NUMPY_SEED = 42
TORCH_SEED = 42

# Device configuration (auto-detect M3 Mac MPS or CUDA or CPU)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Apple M3 GPU
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Data split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# K-fold cross-validation
N_FOLDS = 5

# ============================================================================
# MODEL A: ProtBERT Frozen + SVM
# ============================================================================

PROTBERT_CONFIG = {
    "model_name": "Rostlab/prot_bert",
    "max_length": 512,  # Max sequence length for ProtBERT
    "batch_size": 8,     # Adjust based on available memory
    "embedding_dim": 1024,  # ProtBERT output dimension
    "use_pooling": "mean",  # Options: "mean", "cls", "max"
}

SVM_CONFIG = {
    "kernel": "rbf",
    "C": 1.0,
    "gamma": "scale",
    "probability": True,  # Required for stacking
    "random_state": RANDOM_SEED,
    "cache_size": 1000,  # MB
}

# ============================================================================
# MODEL B: ProtBERT Fine-tuned
# ============================================================================

PROTBERT_FINETUNE_CONFIG = {
    "model_name": "Rostlab/prot_bert",
    "max_length": 512,
    "batch_size": 4,  # Smaller batch for fine-tuning
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 100,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "unfreeze_layers": 2,  # Unfreeze last N encoder layers
    "gradient_accumulation_steps": 2,
}

# ============================================================================
# MODEL C: CNN-BiLSTM
# ============================================================================

CNN_BILSTM_CONFIG = {
    "vocab_size": 25,  # 20 amino acids + special tokens
    "embedding_dim": 128,
    "num_filters": 128,
    "kernel_sizes": [3, 5, 7],  # Multiple kernel sizes
    "lstm_hidden_dim": 128,
    "lstm_num_layers": 2,
    "dropout": 0.3,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "num_epochs": 20,
    "max_seq_length": 1024,
}

# ============================================================================
# MODEL D: Lightweight Transformer
# ============================================================================

LITE_TRANSFORMER_CONFIG = {
    "vocab_size": 25,
    "embedding_dim": 128,
    "d_model": 256,
    "nhead": 4,
    "num_encoder_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "batch_size": 16,
    "learning_rate": 5e-4,
    "num_epochs": 20,
    "max_seq_length": 1024,
}

# ============================================================================
# META-LEARNER (Level 1)
# ============================================================================

META_LEARNER_CONFIG = {
    "model_type": "logistic",  # Options: "logistic", "mlp", "xgboost"
    "C": 1.0,  # For logistic regression
    "penalty": "l2",
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    
    # For MLP meta-learner (if used)
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
}

# ============================================================================
# TRAINING PARAMETERS
# ============================================================================

TRAINING_CONFIG = {
    "early_stopping_patience": 5,
    "min_delta": 1e-4,
    "checkpoint_frequency": 1,  # Save every N epochs
    "gradient_clip_value": 1.0,
    "mixed_precision": False,  # FP16 training (disable for M3)
}

# ============================================================================
# EVALUATION PARAMETERS
# ============================================================================

EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "roc_auc"],
    "threshold": 0.5,  # Classification threshold
    "plot_dpi": 300,  # Figure resolution
    "plot_format": "png",
}

# ============================================================================
# AMINO ACID VOCABULARY
# ============================================================================

# Standard 20 amino acids + special tokens
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<CLS>", "<SEP>", "<MASK>"]

AMINO_ACID_VOCAB = {aa: idx for idx, aa in enumerate(SPECIAL_TOKENS + AMINO_ACIDS)}
VOCAB_SIZE = len(AMINO_ACID_VOCAB)

# Reverse mapping
IDX_TO_AMINO_ACID = {idx: aa for aa, idx in AMINO_ACID_VOCAB.items()}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    "level": "INFO",  # Options: DEBUG, INFO, WARNING, ERROR
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "date_format": "%Y-%m-%d %H:%M:%S",
    "log_file": LOGS_DIR / "training.log",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def ensure_directories() -> None:
    """
    Create all necessary directories if they don't exist.
    
    Should be called at the start of any script.
    """
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR,
        RESULTS_DIR, SAVED_MODELS_DIR, EMBEDDINGS_DIR,
        PREDICTIONS_DIR, METRICS_DIR, LOGS_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_config_summary() -> Dict[str, Any]:
    """
    Get a dictionary summary of all configurations.
    
    Returns:
        Dictionary with all config values
    """
    return {
        "device": str(DEVICE),
        "random_seed": RANDOM_SEED,
        "data_splits": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "models": {
            "protbert_frozen": PROTBERT_CONFIG,
            "protbert_finetune": PROTBERT_FINETUNE_CONFIG,
            "cnn_bilstm": CNN_BILSTM_CONFIG,
            "lite_transformer": LITE_TRANSFORMER_CONFIG,
            "meta_learner": META_LEARNER_CONFIG,
        },
    }


def set_random_seeds() -> None:
    """
    Set random seeds for reproducibility across all libraries.
    """
    import random
    import numpy as np
    
    random.seed(RANDOM_SEED)
    np.random.seed(NUMPY_SEED)
    torch.manual_seed(TORCH_SEED)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
    
    # For MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(TORCH_SEED)


# ============================================================================
# INITIALIZATION
# ============================================================================

# Automatically create directories when config is imported
ensure_directories()

# Set random seeds
set_random_seeds()


# ============================================================================
# VERSION INFO
# ============================================================================

__version__ = "1.0.0"
__author__ = "Tau Stacking Team"
__date__ = "2025-12-18"

if __name__ == "__main__":
    print("=" * 80)
    print("TAU STACKING PROJECT - CONFIGURATION")
    print("=" * 80)
    print(f"\nVersion: {__version__}")
    print(f"Device: {DEVICE}")
    print(f"Root Directory: {ROOT_DIR}")
    print(f"\nAll directories created successfully!")
    print("\nConfiguration Summary:")
    
    config = get_config_summary()
    import json
    print(json.dumps(config, indent=2, default=str))

