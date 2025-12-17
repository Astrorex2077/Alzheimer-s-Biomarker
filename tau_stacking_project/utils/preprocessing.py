# Preprocessing and feature generation
"""
Preprocessing and feature generation utilities.

This module handles:
- ProtBERT embedding generation
- Integer encoding for sequences
- Feature extraction (length, composition, etc.)
- Data normalization and scaling
- Caching embeddings to disk

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.config import (
    PROTBERT_CONFIG,
    EMBEDDINGS_DIR,
    AMINO_ACID_VOCAB,
    DEVICE,
    RANDOM_SEED,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PROTBERT EMBEDDING GENERATION
# ============================================================================

class ProteinSequenceDataset(Dataset):
    """PyTorch Dataset for protein sequences."""
    
    def __init__(self, sequences: List[str], tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            sequences: List of protein sequences
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get tokenized sequence.
        
        Args:
            idx: Index
            
        Returns:
            Dictionary with input_ids, attention_mask
        """
        sequence = self.sequences[idx]
        
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
            'attention_mask': encoded['attention_mask'].squeeze(0)
        }


def compute_protbert_embeddings(
    sequences: List[str],
    model_name: str = PROTBERT_CONFIG['model_name'],
    max_length: int = PROTBERT_CONFIG['max_length'],
    batch_size: int = PROTBERT_CONFIG['batch_size'],
    pooling: str = PROTBERT_CONFIG['use_pooling'],
    device: torch.device = DEVICE,
    cache_path: Optional[Path] = None,
    use_cache: bool = True
) -> np.ndarray:
    """
    Generate ProtBERT embeddings for protein sequences.
    
    Args:
        sequences: List of protein sequences
        model_name: HuggingFace model name
        max_length: Maximum sequence length
        batch_size: Batch size for inference
        pooling: Pooling strategy ('mean', 'cls', or 'max')
        device: Device to use (CPU/CUDA/MPS)
        cache_path: Path to save/load cached embeddings
        use_cache: Whether to use cached embeddings if available
        
    Returns:
        Numpy array of embeddings (n_sequences, embedding_dim)
        
    Raises:
        RuntimeError: If embedding generation fails
    """
    # Check cache
    if use_cache and cache_path and cache_path.exists():
        logger.info(f"Loading cached embeddings from {cache_path}")
        try:
            embeddings = np.load(cache_path)
            if len(embeddings) == len(sequences):
                logger.info(f"Loaded {len(embeddings)} cached embeddings")
                return embeddings
            else:
                logger.warning("Cache size mismatch, regenerating embeddings")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}, regenerating")
    
    logger.info(f"Generating ProtBERT embeddings for {len(sequences)} sequences")
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer and model
        logger.info(f"Loading model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        
        # Create dataset and dataloader
        dataset = ProteinSequenceDataset(sequences, tokenizer, max_length)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0  # Use 0 for M3 MPS compatibility
        )
        
        # Generate embeddings
        all_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Generating embeddings"):
                # Move to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                # Forward pass
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get embeddings based on pooling strategy
                if pooling == 'cls':
                    # Use [CLS] token (first token)
                    embeddings = outputs.last_hidden_state[:, 0, :]
                elif pooling == 'mean':
                    # Mean pooling over sequence length
                    embeddings = torch.sum(
                        outputs.last_hidden_state * attention_mask.unsqueeze(-1),
                        dim=1
                    ) / torch.sum(attention_mask, dim=1, keepdim=True)
                elif pooling == 'max':
                    # Max pooling over sequence length
                    embeddings = torch.max(outputs.last_hidden_state, dim=1)[0]
                else:
                    raise ValueError(f"Invalid pooling strategy: {pooling}")
                
                # Move to CPU and convert to numpy
                embeddings_np = embeddings.cpu().numpy()
                all_embeddings.append(embeddings_np)
        
        # Concatenate all embeddings
        all_embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
        
        # Save to cache
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, all_embeddings)
            logger.info(f"Saved embeddings to cache: {cache_path}")
        
        return all_embeddings
    
    except Exception as e:
        logger.error(f"Error generating ProtBERT embeddings: {e}")
        raise RuntimeError(f"Failed to generate embeddings: {e}")


def generate_and_cache_embeddings_by_split(
    df: pd.DataFrame,
    split_column: str = 'split',
    sequence_column: str = 'sequence',
    output_dir: Path = EMBEDDINGS_DIR
) -> Dict[str, np.ndarray]:
    """
    Generate ProtBERT embeddings for each data split and cache separately.
    
    Args:
        df: DataFrame with sequences and split labels
        split_column: Column name for split labels
        sequence_column: Column name for sequences
        output_dir: Directory to save embeddings
        
    Returns:
        Dictionary mapping split name to embeddings array
    """
    embeddings_dict = {}
    
    for split_name in df[split_column].unique():
        logger.info(f"Processing {split_name} split...")
        
        # Filter sequences for this split
        split_df = df[df[split_column] == split_name]
        sequences = split_df[sequence_column].tolist()
        
        # Generate embeddings
        cache_path = output_dir / f"protbert_{split_name}.npy"
        embeddings = compute_protbert_embeddings(
            sequences,
            cache_path=cache_path,
            use_cache=True
        )
        
        embeddings_dict[split_name] = embeddings
        
        logger.info(f"{split_name}: {embeddings.shape}")
    
    return embeddings_dict


# ============================================================================
# INTEGER ENCODING FOR SEQUENCES
# ============================================================================

def encode_sequences_to_int(
    sequences: List[str],
    vocab: Dict[str, int] = AMINO_ACID_VOCAB,
    max_length: Optional[int] = None,
    padding: str = 'post',
    truncating: str = 'post'
) -> np.ndarray:
    """
    Encode protein sequences to integer arrays.
    
    Args:
        sequences: List of protein sequences
        vocab: Amino acid vocabulary mapping
        max_length: Maximum sequence length (None for automatic)
        padding: Padding strategy ('pre' or 'post')
        truncating: Truncating strategy ('pre' or 'post')
        
    Returns:
        Integer encoded sequences (n_sequences, max_length)
    """
    logger.info(f"Encoding {len(sequences)} sequences to integers")
    
    # Determine max_length if not provided
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
        logger.info(f"Auto-detected max_length: {max_length}")
    
    # Get padding and unknown token indices
    pad_idx = vocab.get('<PAD>', 0)
    unk_idx = vocab.get('<UNK>', 1)
    
    encoded_sequences = []
    
    for seq in sequences:
        # Convert each amino acid to index
        encoded = [vocab.get(aa.upper(), unk_idx) for aa in seq]
        
        # Truncate if necessary
        if len(encoded) > max_length:
            if truncating == 'post':
                encoded = encoded[:max_length]
            else:  # 'pre'
                encoded = encoded[-max_length:]
        
        # Pad if necessary
        if len(encoded) < max_length:
            padding_needed = max_length - len(encoded)
            if padding == 'post':
                encoded = encoded + [pad_idx] * padding_needed
            else:  # 'pre'
                encoded = [pad_idx] * padding_needed + encoded
        
        encoded_sequences.append(encoded)
    
    encoded_array = np.array(encoded_sequences, dtype=np.int32)
    logger.info(f"Encoded shape: {encoded_array.shape}")
    
    return encoded_array


def create_attention_masks(
    encoded_sequences: np.ndarray,
    pad_idx: int = 0
) -> np.ndarray:
    """
    Create attention masks for padded sequences.
    
    Args:
        encoded_sequences: Integer encoded sequences
        pad_idx: Padding token index
        
    Returns:
        Attention masks (1 for real tokens, 0 for padding)
    """
    return (encoded_sequences != pad_idx).astype(np.float32)


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def compute_sequence_features(sequences: List[str]) -> pd.DataFrame:
    """
    Compute hand-crafted features from protein sequences.
    
    Features include:
    - Sequence length
    - Amino acid composition
    - Charged residue percentage
    - Hydrophobic residue percentage
    - Aromatic residue percentage
    
    Args:
        sequences: List of protein sequences
        
    Returns:
        DataFrame with computed features
    """
    logger.info(f"Computing sequence features for {len(sequences)} sequences")
    
    # Amino acid groups
    charged = set('DEKR')
    hydrophobic = set('AVILMFYW')
    aromatic = set('FYW')
    polar = set('STNQ')
    
    features = []
    
    for seq in sequences:
        seq_upper = seq.upper()
        length = len(seq)
        
        if length == 0:
            # Handle empty sequences
            features.append({
                'length': 0,
                'charged_pct': 0,
                'hydrophobic_pct': 0,
                'aromatic_pct': 0,
                'polar_pct': 0,
            })
            continue
        
        # Count amino acids in each group
        charged_count = sum(1 for aa in seq_upper if aa in charged)
        hydrophobic_count = sum(1 for aa in seq_upper if aa in hydrophobic)
        aromatic_count = sum(1 for aa in seq_upper if aa in aromatic)
        polar_count = sum(1 for aa in seq_upper if aa in polar)
        
        features.append({
            'length': length,
            'charged_pct': charged_count / length * 100,
            'hydrophobic_pct': hydrophobic_count / length * 100,
            'aromatic_pct': aromatic_count / length * 100,
            'polar_pct': polar_count / length * 100,
        })
    
    df_features = pd.DataFrame(features)
    logger.info(f"Computed features shape: {df_features.shape}")
    
    return df_features


def compute_amino_acid_composition(sequences: List[str]) -> pd.DataFrame:
    """
    Compute amino acid composition for each sequence.
    
    Args:
        sequences: List of protein sequences
        
    Returns:
        DataFrame with composition percentages for each amino acid
    """
    logger.info(f"Computing amino acid composition for {len(sequences)} sequences")
    
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    composition = []
    
    for seq in sequences:
        seq_upper = seq.upper()
        length = len(seq) if len(seq) > 0 else 1  # Avoid division by zero
        
        comp = {aa: seq_upper.count(aa) / length * 100 for aa in amino_acids}
        composition.append(comp)
    
    df_composition = pd.DataFrame(composition)
    logger.info(f"Composition shape: {df_composition.shape}")
    
    return df_composition


# ============================================================================
# NORMALIZATION AND SCALING
# ============================================================================

def normalize_embeddings(
    embeddings: np.ndarray,
    method: str = 'standardize'
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Normalize embeddings.
    
    Args:
        embeddings: Raw embeddings
        method: Normalization method ('standardize' or 'minmax')
        
    Returns:
        Tuple of (normalized embeddings, fitted scaler)
    """
    logger.info(f"Normalizing embeddings using {method} method")
    
    if method == 'standardize':
        scaler = StandardScaler()
        normalized = scaler.fit_transform(embeddings)
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(embeddings)
    else:
        raise ValueError(f"Invalid normalization method: {method}")
    
    logger.info(f"Normalized embeddings shape: {normalized.shape}")
    
    return normalized, scaler


# ============================================================================
# SAVE/LOAD EMBEDDINGS AND ARRAYS
# ============================================================================

def save_embeddings_and_arrays(
    embeddings_dict: Dict[str, np.ndarray],
    encoded_dict: Optional[Dict[str, np.ndarray]] = None,
    output_dir: Path = EMBEDDINGS_DIR
) -> None:
    """
    Save embeddings and encoded sequences to disk.
    
    Args:
        embeddings_dict: Dictionary of split -> embeddings
        encoded_dict: Dictionary of split -> encoded sequences
        output_dir: Output directory
    """
    logger.info("Saving embeddings and encoded arrays...")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Save embeddings
        for split_name, embeddings in embeddings_dict.items():
            emb_path = output_dir / f"protbert_{split_name}.npy"
            np.save(emb_path, embeddings)
            logger.info(f"Saved {split_name} embeddings: {emb_path}")
        
        # Save encoded sequences
        if encoded_dict:
            for split_name, encoded in encoded_dict.items():
                enc_path = output_dir / f"encoded_{split_name}.npy"
                np.save(enc_path, encoded)
                logger.info(f"Saved {split_name} encoded: {enc_path}")
        
        logger.info("All arrays saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving arrays: {e}")
        raise


def load_embeddings(
    split_name: str,
    embeddings_dir: Path = EMBEDDINGS_DIR
) -> np.ndarray:
    """
    Load embeddings for a specific split.
    
    Args:
        split_name: Name of split ('train', 'val', 'test')
        embeddings_dir: Directory containing embeddings
        
    Returns:
        Embeddings array
        
    Raises:
        FileNotFoundError: If embeddings file doesn't exist
    """
    emb_path = embeddings_dir / f"protbert_{split_name}.npy"
    
    if not emb_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {emb_path}")
    
    logger.info(f"Loading {split_name} embeddings from {emb_path}")
    embeddings = np.load(emb_path)
    logger.info(f"Loaded embeddings shape: {embeddings.shape}")
    
    return embeddings


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing preprocessing utilities."""
    logger.info("=" * 80)
    logger.info("TESTING PREPROCESSING UTILITIES")
    logger.info("=" * 80)
    
    try:
        # Example sequences
        sequences = [
            "ACDEFGHIKLMNPQRSTVWY" * 10,  # 200 aa
            "MAEGEITTFTALTEKFNLEPPTVQPTSVP",
            "ACDEFGHIKLMNPQRSTVWYACDEFGHIKLMNPQRSTVWY",
        ]
        
        logger.info(f"\nTesting with {len(sequences)} sequences")
        
        # Test integer encoding
        logger.info("\n1. Testing integer encoding...")
        encoded = encode_sequences_to_int(sequences, max_length=100)
        logger.info(f"Encoded shape: {encoded.shape}")
        
        # Test attention masks
        masks = create_attention_masks(encoded)
        logger.info(f"Attention masks shape: {masks.shape}")
        
        # Test sequence features
        logger.info("\n2. Testing sequence features...")
        features = compute_sequence_features(sequences)
        logger.info(f"Features:\n{features}")
        
        # Test amino acid composition
        logger.info("\n3. Testing amino acid composition...")
        composition = compute_amino_acid_composition(sequences)
        logger.info(f"Composition shape: {composition.shape}")
        
        # Test ProtBERT embeddings (only if needed)
        logger.info("\n4. Testing ProtBERT embeddings...")
        logger.info("(Skipping for quick test - uncomment to run)")
        # embeddings = compute_protbert_embeddings(sequences[:2], batch_size=2)
        # logger.info(f"Embeddings shape: {embeddings.shape}")
        
        logger.info("\n" + "=" * 80)
        logger.info("PREPROCESSING UTILITIES TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()

