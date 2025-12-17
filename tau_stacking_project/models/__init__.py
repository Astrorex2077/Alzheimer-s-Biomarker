"""
Models package initialization.

This module makes all model classes easily importable.

Author: Senior ML Engineer
Date: 2025-12-18
"""

from models.protbert_frozen import (
    ProtBERTFrozenSVM,
    ProtBERTFrozenDense,
)

from models.protbert_finetune import (
    ProtBERTFineTuneClassifier,
)

from models.cnn_bilstm import (
    CNNBiLSTMClassifier,
)

from models.lite_transformer import (
    LiteTransformerClassifier,
)

from models.meta_learner import (
    MetaLearner,
    LogisticMetaLearner,
    MLPMetaLearner,
)

__version__ = "1.0.0"
__author__ = "Tau Stacking Team"

__all__ = [
    # Model A
    'ProtBERTFrozenSVM',
    'ProtBERTFrozenDense',
    
    # Model B
    'ProtBERTFineTuneClassifier',
    
    # Model C
    'CNNBiLSTMClassifier',
    
    # Model D
    'LiteTransformerClassifier',
    
    # Meta-learner
    'MetaLearner',
    'LogisticMetaLearner',
    'MLPMetaLearner',
]
