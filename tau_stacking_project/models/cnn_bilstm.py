# Model C: CNN-BiLSTM model
"""
Model C: CNN-BiLSTM Classifier

This model uses 1D convolutions to capture local patterns,
followed by bidirectional LSTM for sequential dependencies.

Architecture:
    Embedding -> 1D Conv (multi-kernel) -> BiLSTM -> Dense -> Output

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import (
    CNN_BILSTM_CONFIG,
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
# CNN-BiLSTM CLASSIFIER
# ============================================================================

class CNNBiLSTMClassifier(nn.Module):
    """
    CNN-BiLSTM classifier for protein sequences.
    
    Uses multiple convolutional kernels to capture local motifs,
    then BiLSTM to model long-range dependencies.
    """
    
    def __init__(
        self,
        vocab_size: int = CNN_BILSTM_CONFIG['vocab_size'],
        embedding_dim: int = CNN_BILSTM_CONFIG['embedding_dim'],
        num_filters: int = CNN_BILSTM_CONFIG['num_filters'],
        kernel_sizes: List[int] = CNN_BILSTM_CONFIG['kernel_sizes'],
        lstm_hidden_dim: int = CNN_BILSTM_CONFIG['lstm_hidden_dim'],
        lstm_num_layers: int = CNN_BILSTM_CONFIG['lstm_num_layers'],
        num_classes: int = 2,
        dropout: float = CNN_BILSTM_CONFIG['dropout'],
        padding_idx: int = 0
    ):
        """
        Initialize CNN-BiLSTM classifier.
        
        Args:
            vocab_size: Size of amino acid vocabulary
            embedding_dim: Dimension of amino acid embeddings
            num_filters: Number of filters per kernel size
            kernel_sizes: List of kernel sizes for multi-scale convolution
            lstm_hidden_dim: Hidden dimension of LSTM
            lstm_num_layers: Number of LSTM layers
            num_classes: Number of output classes
            dropout: Dropout rate
            padding_idx: Index for padding token
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_sizes = kernel_sizes
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        
        # Embedding layer
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=padding_idx
        )
        
        # Multiple 1D convolutional layers with different kernel sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2  # Keep sequence length
            )
            for k in kernel_sizes
        ])
        
        # Batch normalization for each conv layer
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(num_filters)
            for _ in kernel_sizes
        ])
        
        # Dropout after convolution
        self.conv_dropout = nn.Dropout(dropout)
        
        # BiLSTM layer
        lstm_input_dim = num_filters * len(kernel_sizes)
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_num_layers > 1 else 0
        )
        
        # Attention layer (optional, improves performance)
        self.attention = nn.Linear(lstm_hidden_dim * 2, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_hidden_dim * 2, 128)
        self.fc_dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Initialized CNN-BiLSTM Classifier")
        logger.info(f"  Embedding: {vocab_size} -> {embedding_dim}")
        logger.info(f"  Conv kernels: {kernel_sizes}")
        logger.info(f"  LSTM: {lstm_num_layers} layers, {lstm_hidden_dim} hidden")
        logger.info(f"  Output: {num_classes} classes")
    
    def _init_weights(self):
        """Initialize weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
    
    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input token IDs (batch_size, seq_len)
            lengths: Actual sequence lengths (batch_size,) - optional
            
        Returns:
            Logits (batch_size, num_classes)
        """
        batch_size, seq_len = x.shape
        
        # Embedding layer
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(x)
        
        # Prepare for convolution: (batch_size, embedding_dim, seq_len)
        embedded_transposed = embedded.transpose(1, 2)
        
        # Apply multiple convolutional layers
        conv_outputs = []
        for conv, bn in zip(self.convs, self.batch_norms):
            # Convolution
            conv_out = conv(embedded_transposed)  # (batch_size, num_filters, seq_len)
            
            # Batch normalization
            conv_out = bn(conv_out)
            
            # ReLU activation
            conv_out = F.relu(conv_out)
            
            conv_outputs.append(conv_out)
        
        # Concatenate outputs from different kernel sizes
        # (batch_size, num_filters * len(kernel_sizes), seq_len)
        conv_concat = torch.cat(conv_outputs, dim=1)
        
        # Transpose back for LSTM: (batch_size, seq_len, num_filters * len(kernel_sizes))
        conv_features = conv_concat.transpose(1, 2)
        
        # Dropout after convolution
        conv_features = self.conv_dropout(conv_features)
        
        # BiLSTM layer
        if lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                conv_features,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False
            )
            lstm_out, (hidden, cell) = self.lstm(packed)
            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out,
                batch_first=True,
                total_length=seq_len
            )
        else:
            # (batch_size, seq_len, lstm_hidden_dim * 2)
            lstm_out, (hidden, cell) = self.lstm(conv_features)
        
        # Attention mechanism
        # (batch_size, seq_len, 1)
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        
        # Apply attention
        # (batch_size, lstm_hidden_dim * 2)
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Fully connected layers
        fc1_out = F.relu(self.fc1(attended))
        fc1_out = self.fc_dropout(fc1_out)
        logits = self.fc2(fc1_out)
        
        return logits
    
    def predict_proba(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Input token IDs
            lengths: Actual sequence lengths
            
        Returns:
            Probabilities (batch_size, num_classes)
        """
        logits = self.forward(x, lengths)
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
                'num_filters': self.num_filters,
                'kernel_sizes': self.kernel_sizes,
                'lstm_hidden_dim': self.lstm_hidden_dim,
                'lstm_num_layers': self.lstm_num_layers,
                'num_classes': self.num_classes,
                'dropout': self.dropout_rate,
            }
        }, save_path)
        
        logger.info(f"Model saved: {save_path}")
    
    @classmethod
    def load(
        cls,
        load_path: Path,
        device: torch.device = DEVICE
    ) -> 'CNNBiLSTMClassifier':
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
            num_filters=config['num_filters'],
            kernel_sizes=config['kernel_sizes'],
            lstm_hidden_dim=config['lstm_hidden_dim'],
            lstm_num_layers=config['lstm_num_layers'],
            num_classes=config['num_classes'],
            dropout=config['dropout']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model


# ============================================================================
# DATASET FOR CNN-BiLSTM
# ============================================================================

class ProteinSequenceDataset(torch.utils.data.Dataset):
    """Dataset for CNN-BiLSTM training."""
    
    def __init__(
        self,
        encoded_sequences: torch.Tensor,
        labels: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ):
        """
        Initialize dataset.
        
        Args:
            encoded_sequences: Integer-encoded sequences (n_samples, max_len)
            labels: Labels (n_samples,)
            lengths: Actual sequence lengths (n_samples,) - optional
        """
        self.encoded_sequences = encoded_sequences
        self.labels = labels
        self.lengths = lengths
    
    def __len__(self) -> int:
        return len(self.encoded_sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with sequence, label, and optionally length
        """
        item = {
            'sequence': self.encoded_sequences[idx],
            'label': self.labels[idx]
        }
        
        if self.lengths is not None:
            item['length'] = self.lengths[idx]
        
        return item


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing Model C."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL C: CNN-BiLSTM")
    logger.info("=" * 80)
    
    # Create model
    logger.info("\n1. Creating model...")
    model = CNNBiLSTMClassifier(
        vocab_size=25,
        embedding_dim=128,
        num_filters=128,
        kernel_sizes=[3, 5, 7],
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        num_classes=2,
        dropout=0.3
    )
    
    logger.info(f"Total parameters: {model.get_trainable_parameters():,}")
    
    # Test forward pass
    logger.info("\n2. Testing forward pass...")
    batch_size = 4
    seq_length = 100
    
    # Create dummy data
    x = torch.randint(0, 25, (batch_size, seq_length))
    lengths = torch.tensor([100, 80, 60, 90])
    
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
        probs = model.predict_proba(x, lengths)
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Probabilities shape: {probs.shape}")
    logger.info(f"Probabilities sum to 1: {torch.allclose(probs.sum(dim=1), torch.ones(batch_size))}")
    
    # Test without lengths
    logger.info("\n3. Testing without length masking...")
    with torch.no_grad():
        logits_no_length = model(x)
    logger.info(f"Logits shape (no length): {logits_no_length.shape}")
    
    # Test save/load
    logger.info("\n4. Testing save/load...")
    save_path = SAVED_MODELS_DIR / "test_cnn_bilstm.pth"
    model.save(save_path)
    model_loaded = CNNBiLSTMClassifier.load(save_path)
    logger.info("Save/load test passed")
    
    # Test that loaded model gives same output
    model_loaded.eval()
    with torch.no_grad():
        logits_loaded = model_loaded(x, lengths)
    
    logger.info(f"Outputs match: {torch.allclose(logits, logits_loaded, atol=1e-6)}")
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL C TESTS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

