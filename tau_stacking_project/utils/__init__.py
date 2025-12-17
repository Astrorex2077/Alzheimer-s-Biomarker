"""
Utils package initialization.

This module makes all utility functions easily importable.

Author: Senior ML Engineer
Date: 2025-12-18
"""

from utils.config import (
    # Paths
    ROOT_DIR,
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    SAVED_MODELS_DIR,
    EMBEDDINGS_DIR,
    PREDICTIONS_DIR,
    METRICS_DIR,
    LOGS_DIR,
    
    # Config
    DEVICE,
    RANDOM_SEED,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    N_FOLDS,
    
    # Model configs
    PROTBERT_CONFIG,
    SVM_CONFIG,
    PROTBERT_FINETUNE_CONFIG,
    CNN_BILSTM_CONFIG,
    LITE_TRANSFORMER_CONFIG,
    META_LEARNER_CONFIG,
    
    # Amino acid vocab
    AMINO_ACID_VOCAB,
    VOCAB_SIZE,
    
    # Helper functions
    ensure_directories,
    get_config_summary,
    set_random_seeds,
)

from utils.dataset import (
    load_fasta,
    save_fasta,
    load_labels,
    create_synthetic_labels,
    make_splits,
    make_kfold_splits,
    validate_sequences,
    save_core_tables,
    load_core_tables,
)

from utils.preprocessing import (
    compute_protbert_embeddings,
    generate_and_cache_embeddings_by_split,
    encode_sequences_to_int,
    create_attention_masks,
    compute_sequence_features,
    compute_amino_acid_composition,
    normalize_embeddings,
    save_embeddings_and_arrays,
    load_embeddings,
)

from utils.training_loops import (
    EarlyStopping,
    train_torch_model,
    train_sklearn_model,
    predict_with_torch_model,
    predict_with_sklearn_model,
    save_checkpoint,
    load_checkpoint,
)

from utils.evaluation import (
    compute_classification_metrics,
    compute_metrics_by_class,
    plot_roc_curve,
    plot_multiple_roc_curves,
    plot_precision_recall_curve,
    plot_confusion_matrix,
    plot_accuracy_bar,
    create_metrics_comparison_table,
    export_predictions,
    save_metrics_json,
)

from utils.stacking_pipeline import (
    build_meta_features,
    load_validation_predictions,
    save_base_model_predictions,
    train_meta_learner,
    StackingEnsemble,
)

__version__ = "1.0.0"
__author__ = "Tau Stacking Team"

__all__ = [
    # Config
    'ROOT_DIR',
    'DATA_DIR',
    'RAW_DATA_DIR',
    'PROCESSED_DATA_DIR',
    'MODELS_DIR',
    'RESULTS_DIR',
    'SAVED_MODELS_DIR',
    'EMBEDDINGS_DIR',
    'PREDICTIONS_DIR',
    'METRICS_DIR',
    'LOGS_DIR',
    'DEVICE',
    'RANDOM_SEED',
    'TRAIN_RATIO',
    'VAL_RATIO',
    'TEST_RATIO',
    'N_FOLDS',
    'PROTBERT_CONFIG',
    'SVM_CONFIG',
    'PROTBERT_FINETUNE_CONFIG',
    'CNN_BILSTM_CONFIG',
    'LITE_TRANSFORMER_CONFIG',
    'META_LEARNER_CONFIG',
    'AMINO_ACID_VOCAB',
    'VOCAB_SIZE',
    'ensure_directories',
    'get_config_summary',
    'set_random_seeds',
    
    # Dataset
    'load_fasta',
    'save_fasta',
    'load_labels',
    'create_synthetic_labels',
    'make_splits',
    'make_kfold_splits',
    'validate_sequences',
    'save_core_tables',
    'load_core_tables',
    
    # Preprocessing
    'compute_protbert_embeddings',
    'generate_and_cache_embeddings_by_split',
    'encode_sequences_to_int',
    'create_attention_masks',
    'compute_sequence_features',
    'compute_amino_acid_composition',
    'normalize_embeddings',
    'save_embeddings_and_arrays',
    'load_embeddings',
    
    # Training
    'EarlyStopping',
    'train_torch_model',
    'train_sklearn_model',
    'predict_with_torch_model',
    'predict_with_sklearn_model',
    'save_checkpoint',
    'load_checkpoint',
    
    # Evaluation
    'compute_classification_metrics',
    'compute_metrics_by_class',
    'plot_roc_curve',
    'plot_multiple_roc_curves',
    'plot_precision_recall_curve',
    'plot_confusion_matrix',
    'plot_accuracy_bar',
    'create_metrics_comparison_table',
    'export_predictions',
    'save_metrics_json',
    
    # Stacking
    'build_meta_features',
    'load_validation_predictions',
    'save_base_model_predictions',
    'train_meta_learner',
    'StackingEnsemble',
]
