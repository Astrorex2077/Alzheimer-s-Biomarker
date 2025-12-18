# Model B: Fine-tuned ProtBERT classifier
"""
Model B: Fine-tuned ProtBERT Classifier

This model fine-tunes the last few layers of ProtBERT
along with a classification head for task-specific learning.

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


from utils.config import (
    PROTBERT_FINETUNE_CONFIG,
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
# PROTBERT FINE-TUNE CLASSIFIER
# ============================================================================

class ProtBERTFineTuneClassifier(nn.Module):
    """
    Fine-tunable ProtBERT classifier.
    
    Unfreezes the last N encoder layers of ProtBERT and adds
    a classification head. More flexible than frozen approach.
    """
    
    def __init__(
        self,
        model_name: str = PROTBERT_FINETUNE_CONFIG['model_name'],
        num_classes: int = 2,
        dropout: float = PROTBERT_FINETUNE_CONFIG['dropout'],
        unfreeze_layers: int = PROTBERT_FINETUNE_CONFIG['unfreeze_layers'],
        pooling: str = 'cls'
    ):
        """
        Initialize fine-tunable ProtBERT classifier.
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            dropout: Dropout rate
            unfreeze_layers: Number of encoder layers to unfreeze (from end)
            pooling: Pooling strategy ('cls' or 'mean')
        """
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.unfreeze_layers = unfreeze_layers
        self.pooling = pooling
        
        # Load ProtBERT
        logger.info(f"Loading ProtBERT model: {model_name}")
        self.protbert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        
        # Get embedding dimension
        self.hidden_size = self.protbert.config.hidden_size
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Freeze all ProtBERT layers initially
        self._freeze_protbert()
        
        # Unfreeze last N encoder layers
        self._unfreeze_last_layers(unfreeze_layers)
        
        logger.info(f"Initialized ProtBERT Fine-tune Classifier")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Unfrozen layers: {unfreeze_layers}")
        logger.info(f"  Pooling: {pooling}")
    
    def _freeze_protbert(self) -> None:
        """Freeze all ProtBERT parameters."""
        for param in self.protbert.parameters():
            param.requires_grad = False
    
    def _unfreeze_last_layers(self, n_layers: int) -> None:
        """
        Unfreeze last N encoder layers.
        
        Args:
            n_layers: Number of layers to unfreeze from end
        """
        if n_layers <= 0:
            logger.info("All ProtBERT layers remain frozen")
            return
        
        # ProtBERT has encoder layers
        encoder_layers = self.protbert.encoder.layer
        total_layers = len(encoder_layers)
        
        # Unfreeze last n_layers
        layers_to_unfreeze = encoder_layers[-n_layers:]
        
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        
        logger.info(f"Unfroze last {n_layers} encoder layers (out of {total_layers})")
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get ProtBERT outputs
        outputs = self.protbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Pool sequence representations
        if self.pooling == 'cls':
            # Use [CLS] token (first token)
            pooled = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'mean':
            # Mean pooling over sequence
            last_hidden = outputs.last_hidden_state
            pooled = torch.sum(
                last_hidden * attention_mask.unsqueeze(-1),
                dim=1
            ) / torch.sum(attention_mask, dim=1, keepdim=True)
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")
        
        # Classification head
        logits = self.classifier(pooled)
        
        return logits
    
    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Probabilities (batch_size, num_classes)
        """
        logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=1)
    
    def get_trainable_parameters(self) -> int:
        """
        Count trainable parameters.
        
        Returns:
            Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_optimizer(
        self,
        learning_rate: float = PROTBERT_FINETUNE_CONFIG['learning_rate'],
        weight_decay: float = PROTBERT_FINETUNE_CONFIG['weight_decay']
    ) -> AdamW:
        """
        Get optimizer with layer-wise learning rates.
        
        Args:
            learning_rate: Learning rate
            weight_decay: Weight decay
            
        Returns:
            AdamW optimizer
        """
        # Different learning rates for ProtBERT and classifier
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.protbert.named_parameters() if p.requires_grad],
                'lr': learning_rate * 0.1,  # Lower LR for pre-trained layers
                'weight_decay': weight_decay
            },
            {
                'params': self.classifier.parameters(),
                'lr': learning_rate,  # Higher LR for new classifier
                'weight_decay': weight_decay
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        logger.info(f"Optimizer created with LR={learning_rate}")
        
        return optimizer
    
    def get_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        num_warmup_steps: Optional[int] = None
    ):
        """
        Get learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            num_training_steps: Total training steps
            num_warmup_steps: Warmup steps (default: 10% of total)
            
        Returns:
            Learning rate scheduler
        """
        if num_warmup_steps is None:
            num_warmup_steps = int(0.1 * num_training_steps)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        logger.info(f"Scheduler created: {num_warmup_steps} warmup, {num_training_steps} total steps")
        
        return scheduler
    
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
                'model_name': self.model_name,
                'num_classes': self.num_classes,
                'dropout': self.dropout_rate,
                'unfreeze_layers': self.unfreeze_layers,
                'pooling': self.pooling,
            }
        }, save_path)
        
        logger.info(f"Model saved: {save_path}")
    
    @classmethod
    def load(
        cls,
        load_path: Path,
        device: torch.device = DEVICE
    ) -> 'ProtBERTFineTuneClassifier':
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
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            dropout=config['dropout'],
            unfreeze_layers=config['unfreeze_layers'],
            pooling=config['pooling']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        return model


# ============================================================================
# DATASET FOR FINE-TUNING
# ============================================================================

class ProteinSequenceDataset(torch.utils.data.Dataset):
    """Dataset for ProtBERT fine-tuning."""
    
    def __init__(
        self,
        sequences: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = PROTBERT_FINETUNE_CONFIG['max_length']
    ):
        """
        Initialize dataset.
        
        Args:
            sequences: List of protein sequences
            labels: List of labels
            tokenizer: ProtBERT tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get item.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with input_ids, attention_mask, label
        """
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Add spaces between amino acids (ProtBERT requirement)
        spaced_sequence = " ".join(list(sequence[:self.max_length]))
        
        # Tokenize
        encoded = self.tokenizer(
            spaced_sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing Model B."""
    logger.info("=" * 80)
    logger.info("TESTING MODEL B: PROTBERT FINE-TUNE")
    logger.info("=" * 80)
    
    # Create model
    logger.info("\n1. Creating model...")
    model = ProtBERTFineTuneClassifier(
        num_classes=2,
        dropout=0.1,
        unfreeze_layers=2,
        pooling='cls'
    )
    
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Trainable parameters: {model.get_trainable_parameters():,}")
    
    # Test forward pass
    logger.info("\n2. Testing forward pass...")
    batch_size = 4
    seq_length = 128
    
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = model.predict_proba(input_ids, attention_mask)
    
    logger.info(f"Logits shape: {logits.shape}")
    logger.info(f"Probabilities shape: {probs.shape}")
    
    # Test optimizer
    logger.info("\n3. Testing optimizer...")
    optimizer = model.get_optimizer()
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    
    # Test save/load
    logger.info("\n4. Testing save/load...")
    save_path = SAVED_MODELS_DIR / "test_protbert_finetune.pth"
    model.save(save_path)
    model_loaded = ProtBERTFineTuneClassifier.load(save_path)
    logger.info("Save/load test passed")
    
    logger.info("\n" + "=" * 80)
    logger.info("MODEL B TESTS COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

