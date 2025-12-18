# Model D: Lightweight Transformer encoder
"""
Model D: Lightweight Transformer Encoder

This model uses a compact transformer architecture with fewer layers
and smaller dimensions than full-scale transformers like ProtBERT.

Architecture:
    Embedding -> Positional Encoding -> Transformer Encoder -> Pooling -> Dense

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import math

from utils.config import (
    LITE_TRANSFORMER_CONFIG,
    AMINO_ACID_VOCAB,
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
# POSITIONAL ENCODING
# ============================================================================

class PositionalEncoding(nn.Module):
    """
    Positional encoding using sine and cosine functions.
    
    Adds position information to embeddings so the transformer
    can understand sequence order.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Dimension of embeddings
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input embeddings (batch_size, seq_len, d_model)
            
        Returns:
            Embeddings with positional encoding
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ============================================================================
# LITE TRANSFORMER CLASSIFIER
# ============================================================================

class LiteTransformerClassifier(nn.Module):
    """
    Lightweight Transformer classifier for protein sequences.
    
    Uses a compact transformer encoder with fewer layers and
    smaller hidden dimensions for efficient training.
    """
    
    def __init__(
        self,
        vocab_size: int = LITE_TRANSFORMER_CONFIG['vocab_size'],
        embedding_dim: int = LITE_TRANSFORMER_CONFIG['embedding_dim'],
        d_model: int = LITE_TRANSFORMER_CONFIG['d_model'],
        nhead: int = LITE_TRANSFORMER_CONFIG['nhead'],
        num_encoder_layers: int = LITE_TRANSFORMER_CONFIG['num_encoder_layers'],
        dim_feedforward: int = LITE_TRANSFORMER_CONFIG['dim_feedforward'],
        num_classes: int = 2,
        dropout: float = LITE_TRANSFORMER_CONFIG['dropout'],
        max_seq_length: int = LITE_TRANSFORMER_CONFIG['max_seq_length'],
        padding_idx: int = 0,
        use_cls_token: bool = True
    ):
        """
        Initialize Lite Transformer classifier.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Initial embedding dimension
            d_model: Transformer hidden dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            num_classes: Number of output classes
            dropout: Dropout rate
            max_seq_length: Maximum sequence length
            padding_idx: Index for padding token
        """
        super().__init__()
        self.use_cls_token = use_cls_token
    
        # ⭐ FIX: Account for [CLS] token in positional encoding
        actual_max_length = max_seq_length + 1 if use_cls_token else max_seq_length
    
        # Positional encoding
        self.pos_encoder = nn.Parameter(
        torch.randn(1, actual_max_length, d_model)  # ⭐ Use actual_max_length
    )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.max_seq_length = max_seq_length
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Project embedding to d_model if different
        if embedding_dim != d_model:
            self.embedding_projection = nn.Linear(embedding_dim, d_model)
        else:
            self.embedding_projection = nn.Identity()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_length,
            dropout=dropout
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Pooling strategy: we'll use both [CLS] token and mean pooling
        self.use_cls_token = True
        if self.use_cls_token:
            # Add a learnable [CLS] token embedding
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized Lite Transformer Classifier")
        logger.info(f"  Embedding: {vocab_size} -> {embedding_dim}")
        logger.info(f"  Transformer: d_model={d_model}, heads={nhead}, layers={num_encoder_layers}")
        logger.info(f"  Feedforward: {dim_feedforward}")
        logger.info(f"  Output: {num_classes} classes")
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize embeddings
        nn.init.normal_(self.embedding.weight, mean=0, std=0.02)
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, mean=0, std=0.02)
        
        # Initialize linear layers in classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _create_padding_mask(self, x: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
        """
        Create padding mask for transformer.
        
        Args:
            x: Input token IDs (batch_size, seq_len)
            padding_idx: Padding token index
            
        Returns:
            Mask (batch_size, seq_len) where True indicates padding
        """
        return x == padding_idx
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token IDs (batch_size, seq_len)
            mask: Optional padding mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # Create padding mask if not provided
        if mask is None:
            mask = self._create_padding_mask(x)
        
        # Embedding layer
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Project to d_model
        # (batch_size, seq_len, d_model)
        embedded = self.embedding_projection(embedded)
        
        # Add [CLS] token at the beginning if using
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_model)
            embedded = torch.cat([cls_tokens, embedded], dim=1)  # (batch_size, seq_len+1, d_model)
            
            # Extend mask for [CLS] token (not masked)
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
            mask = torch.cat([cls_mask, mask], dim=1)  # (batch_size, seq_len+1)
        
        # Add positional encoding
        embedded = self.pos_encoder(embedded)
        
        # Transformer encoder
        # src_key_padding_mask: (batch_size, seq_len) where True = ignore
        encoded = self.transformer_encoder(
            embedded,
            src_key_padding_mask=mask
        )
        
        # Pooling
        if self.use_cls_token:
            # Use [CLS] token representation
            pooled = encoded[:, 0, :]  # (batch_size, d_model)
        else:
            # Mean pooling over non-padded tokens
            mask_expanded = mask.unsqueeze(-1).expand(encoded.size())
            sum_encoded = torch.sum(encoded * (~mask_expanded), dim=1)
            sum_mask = torch.sum(~mask, dim=1, keepdim=True)
            pooled = sum_encoded / sum_mask  # (batch_size, d_model)
        
        # Classification head
        logits = self.classifier(pooled)
        
        return logits
    
    def predict_proba(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input token IDs
            mask: Optional padding mask
            
        Returns:
            Probabilities (batch_size, num_classes)
        """
        logits = self.forward(x, mask)
        return torch.softmax(logits, dim=1)
    
    def get_trainable_parameters(self) -> int:
        """
        Count trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
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
                'vocab_size': self.vocab_size,
                'embedding_dim': self.embedding_dim,
                'd_model': self.d_model,
                'nhead': self.nhead,
                'num_encoder_layers': self.num_encoder_layers,
                'dim_feedforward': self.dim_feedforward,
                'num_classes': self.num_classes,
                'dropout': self.dropout_rate,
                'max_seq_length': self.max_seq_length,
            }
        }, save_path)
        
        logger.info(f"Model saved: {save_path}")
    
    @classmethod
    def load(
        cls,
        load_path: Path,
        device: torch.device = DEVICE
    ) -> 'LiteTransformerClassifier':
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
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            d_model=config['d_model'],
            nhead=config['nhead'],
            num_encoder_layers=config['num_encoder_layers'],
            dim_feedforward=config['dim_feedforward'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            max_seq_length=config['max_seq_length']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model


# ============================================================================
# DATASET FOR LITE TRANSFORMER
# ============================================================================

class ProteinSequenceDataset(torch.utils.data.Dataset):
    """Dataset for Lite Transformer training."""
    
    def __init__(
        self,
        encoded_sequences: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Initialize dataset.
        
        Args:
            encoded_sequences: Integer-encoded sequences (n_samples, max_len)
            labels: Labels (n_samples,)
        """
        self.encoded_sequences = encoded_sequences
        self.labels = labels
    
    def __len__(self) -> int:
        return len(self.encoded_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with sequence and label
        """
        return {
            'sequence': self.encoded_sequences[idx],
            'label': self.labels[idx]
        }


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing Model D."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL D: LITE TRANSFORMER")
    logger.info("=" * 80)
    
    # Create model
    logger.info("\n1. Creating model...")
    model = LiteTransformerClassifier(
        vocab_size=25,
        embedding_dim=128,
        d_model=256,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        num_classes=2,
        dropout=0.1,
        max_seq_length=1024
    )
    
    logger.info(f"Total parameters: {model.get_trainable_parameters():,}")
    
    # Test forward pass
    logger.info("\n2. Testing forward pass...")
    batch_size = 4
    seq_length = 100
    
    # Create dummy data (with some padding)
    x = torch.randint(1, 25, (batch_size, seq_length))
    # Add some padding at the end
    x[:, 80:] = 0  # Padding token
    
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = model.predict_proba(x)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Probabilities shape: {probs.shape}")
    logger.info(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(batch_size))}")
    
    # Test with explicit mask
    logger.info("\n3. Testing with explicit mask...")
    mask = (x == 0)  # Padding mask
    with torch.no_grad():
        logits_with_mask = model(x, mask)
    logger.info(f"Outputs match: {torch.allclose(logits, logits_with_mask, atol=1e-6)}")
    
    # Test attention heads compatibility
    logger.info("\n4. Testing attention configuration...")
    logger.info(f"d_model={model.d_model}, nhead={model.nhead}")
    logger.info(f"d_model % nhead = {model.d_model % model.nhead} (should be 0)")
    
    # Test save/load
    logger.info("\n5. Testing save/load...")
    save_path = SAVED_MODELS_DIR / "test_lite_transformer.pth"
    model.save(save_path)
    model_loaded = LiteTransformerClassifier.load(save_path)
    logger.info("Save/load test passed")
    
    # Test that loaded model gives same output
    model_loaded.eval()
    with torch.no_grad():
        logits_loaded = model_loaded(x)
    
    logger.info(f"Outputs match after load: {torch.allclose(logits, logits_loaded, atol=1e-6)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL D TESTS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

