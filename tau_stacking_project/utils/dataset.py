# Dataset loading and splitting utilities
"""
Dataset loading and splitting utilities for tau protein sequences.

This module handles:
- Loading FASTA files
- Creating/loading labels
- Train/val/test splitting with stratification
- K-fold cross-validation setup
- Data validation and quality checks

Author: Senior ML Engineer
Date: 2025-12-18
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.model_selection import train_test_split, StratifiedKFold

from utils.config import (
    FASTA_FILE,
    SEQUENCES_CSV,
    LABELS_CSV,
    SPLITS_CSV,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    N_FOLDS,
    RANDOM_SEED,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# FASTA FILE OPERATIONS
# ============================================================================

def load_fasta(fasta_path):
    """
    Load sequences from FASTA file with improved parsing.
    
    Args:
        fasta_path: Path to FASTA file
    
    Returns:
        pandas DataFrame with columns: protein_id, sequence, description
    """
    from Bio import SeqIO
    import pandas as pd
    from pathlib import Path
    
    fasta_path = Path(fasta_path)
    
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    
    sequences = []
    
    try:
        # Parse FASTA file using BioPython
        for record in SeqIO.parse(str(fasta_path), "fasta"):
            sequences.append({
                'protein_id': record.id,
                'sequence': str(record.seq),
                'description': record.description
            })
    except Exception as e:
        print(f"⚠️ BioPython parsing failed: {e}")
        print("Trying manual parsing...")
        
        # Fallback: Manual parsing
        with open(fasta_path, 'r') as f:
            current_id = None
            current_seq = []
            current_desc = None
            
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                if line.startswith('>'):
                    # Save previous sequence
                    if current_id is not None:
                        sequences.append({
                            'protein_id': current_id,
                            'sequence': ''.join(current_seq),
                            'description': current_desc
                        })
                    
                    # Start new sequence
                    current_desc = line[1:]  # Remove '>'
                    current_id = current_desc.split()[0] if current_desc else f"SEQ_{len(sequences)+1}"
                    current_seq = []
                else:
                    # Add to current sequence
                    current_seq.append(line)
            
            # Don't forget the last sequence
            if current_id is not None:
                sequences.append({
                    'protein_id': current_id,
                    'sequence': ''.join(current_seq),
                    'description': current_desc
                })
    
    if not sequences:
        raise ValueError(f"No sequences found in {fasta_path}")
    
    df = pd.DataFrame(sequences)
    
    print(f"✅ Loaded {len(df)} sequences from {fasta_path.name}")
    print(f"   Average length: {df['sequence'].str.len().mean():.1f} amino acids")
    
    return df



def save_fasta(df: pd.DataFrame, output_path: Union[str, Path], 
               id_column: str = 'protein_id', 
               seq_column: str = 'sequence') -> None:
    """
    Save sequences from DataFrame to FASTA file.
    
    Args:
        df: DataFrame containing sequences
        output_path: Output FASTA file path
        id_column: Column name for sequence IDs
        seq_column: Column name for sequences
        
    Raises:
        ValueError: If required columns are missing
    """
    output_path = Path(output_path)
    
    if id_column not in df.columns or seq_column not in df.columns:
        raise ValueError(f"DataFrame must contain '{id_column}' and '{seq_column}' columns")
    
    logger.info(f"Saving {len(df)} sequences to {output_path}")
    
    try:
        records = []
        for _, row in df.iterrows():
            record = SeqRecord(
                Seq(row[seq_column]),
                id=row[id_column],
                description=row.get('description', '')
            )
            records.append(record)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(records, output_path, "fasta")
        logger.info(f"Successfully saved FASTA file: {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving FASTA file: {e}")
        raise


# ============================================================================
# LABEL OPERATIONS
# ============================================================================

def load_labels(labels_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load labels from CSV file.
    
    Expected format:
        protein_id,label,source
        P10636,1,literature
        Q12345,0,assumed
    
    Args:
        labels_path: Path to labels CSV file
        
    Returns:
        DataFrame with columns: ['protein_id', 'label', 'source']
        
    Raises:
        FileNotFoundError: If labels file doesn't exist
        ValueError: If required columns are missing
    """
    labels_path = Path(labels_path)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    logger.info(f"Loading labels from: {labels_path}")
    
    try:
        df = pd.read_csv(labels_path)
        
        # Validate required columns
        required_cols = ['protein_id', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate label values
        valid_labels = {0, 1}
        invalid = set(df['label'].unique()) - valid_labels
        if invalid:
            raise ValueError(f"Invalid label values: {invalid}. Must be 0 or 1")
        
        logger.info(f"Loaded {len(df)} labels")
        logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error loading labels: {e}")
        raise


def create_synthetic_labels(df_sequences: pd.DataFrame, 
                           positive_ratio: float = 0.3) -> pd.DataFrame:
    """
    Create synthetic labels for demonstration/testing purposes.
    
    WARNING: These are random labels for testing only!
    Replace with real labels for actual research.
    
    Args:
        df_sequences: DataFrame with protein sequences
        positive_ratio: Ratio of positive (misfolding) examples
        
    Returns:
        DataFrame with columns: ['protein_id', 'label', 'source']
    """
    logger.warning("Creating SYNTHETIC labels - replace with real labels for research!")
    
    n_samples = len(df_sequences)
    n_positive = int(n_samples * positive_ratio)
    
    # Create labels (random but reproducible)
    np.random.seed(RANDOM_SEED)
    labels = np.concatenate([
        np.ones(n_positive, dtype=int),
        np.zeros(n_samples - n_positive, dtype=int)
    ])
    np.random.shuffle(labels)
    
    df_labels = pd.DataFrame({
        'protein_id': df_sequences['protein_id'].values,
        'label': labels,
        'source': 'synthetic'
    })
    
    logger.info(f"Created {len(df_labels)} synthetic labels")
    logger.info(f"Label distribution: {df_labels['label'].value_counts().to_dict()}")
    
    return df_labels


# ============================================================================
# DATA SPLITTING
# ============================================================================

def make_splits(df: pd.DataFrame,
                train_ratio: float = TRAIN_RATIO,
                val_ratio: float = VAL_RATIO,
                test_ratio: float = TEST_RATIO,
                stratify_column: Optional[str] = 'label',
                random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Split dataset into train/val/test sets with stratification.
    
    Args:
        df: DataFrame with sequences and labels
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        stratify_column: Column to use for stratification (None for random)
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with added 'split' column ('train', 'val', or 'test')
        
    Raises:
        ValueError: If ratios don't sum to 1 or invalid stratify_column
    """
    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")
    
    if stratify_column and stratify_column not in df.columns:
        raise ValueError(f"Stratify column '{stratify_column}' not found in DataFrame")
    
    logger.info(f"Creating splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
    
    df_split = df.copy()
    
    # Prepare stratification
    stratify = df[stratify_column] if stratify_column else None
    
    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        df.index,
        test_size=test_ratio,
        stratify=stratify,
        random_state=random_state
    )
    
    # Second split: separate train and val from remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    if stratify_column:
        stratify_train_val = df.loc[train_val_idx, stratify_column]
    else:
        stratify_train_val = None
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_adjusted,
        stratify=stratify_train_val,
        random_state=random_state
    )
    
    # Assign split labels
    df_split['split'] = 'unknown'
    df_split.loc[train_idx, 'split'] = 'train'
    df_split.loc[val_idx, 'split'] = 'val'
    df_split.loc[test_idx, 'split'] = 'test'
    
    # Log split statistics
    logger.info("Split statistics:")
    for split_name in ['train', 'val', 'test']:
        split_data = df_split[df_split['split'] == split_name]
        logger.info(f"  {split_name}: {len(split_data)} samples")
        if stratify_column:
            dist = split_data[stratify_column].value_counts().to_dict()
            logger.info(f"    Label distribution: {dist}")
    
    return df_split


def make_kfold_splits(df: pd.DataFrame,
                      n_splits: int = N_FOLDS,
                      stratify_column: Optional[str] = 'label',
                      random_state: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Create K-fold cross-validation splits.
    
    Args:
        df: DataFrame with sequences and labels
        n_splits: Number of folds
        stratify_column: Column for stratified K-fold
        random_state: Random seed
        
    Returns:
        DataFrame with 'fold' column (0 to n_splits-1)
    """
    logger.info(f"Creating {n_splits}-fold splits")
    
    df_fold = df.copy()
    df_fold['fold'] = -1
    
    if stratify_column and stratify_column in df.columns:
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y = df[stratify_column]
    else:
        from sklearn.model_selection import KFold
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        y = None
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(df, y)):
        df_fold.iloc[val_idx, df_fold.columns.get_loc('fold')] = fold
    
    logger.info(f"Fold distribution: {df_fold['fold'].value_counts().sort_index().to_dict()}")
    
    return df_fold


# ============================================================================
# DATA VALIDATION
# ============================================================================

def validate_sequences(df: pd.DataFrame, 
                       sequence_column: str = 'sequence') -> Tuple[bool, List[str]]:
    """
    Validate protein sequences for correctness.
    
    Args:
        df: DataFrame with sequences
        sequence_column: Column name containing sequences
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check for valid amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
    
    for idx, row in df.iterrows():
        seq = row[sequence_column]
        protein_id = row.get('protein_id', idx)
        
        # Check empty sequences
        if not seq or len(seq) == 0:
            errors.append(f"{protein_id}: Empty sequence")
            continue
        
        # Check for invalid characters
        invalid_chars = set(seq.upper()) - valid_aa
        if invalid_chars:
            errors.append(f"{protein_id}: Invalid amino acids: {invalid_chars}")
        
        # Check for extremely short sequences
        if len(seq) < 10:
            errors.append(f"{protein_id}: Sequence too short ({len(seq)} aa)")
        
        # Check for extremely long sequences (likely error)
        if len(seq) > 5000:
            errors.append(f"{protein_id}: Sequence suspiciously long ({len(seq)} aa)")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info("All sequences validated successfully")
    else:
        logger.warning(f"Found {len(errors)} validation errors")
    
    return is_valid, errors


# ============================================================================
# SAVE/LOAD CORE TABLES
# ============================================================================

def save_core_tables(df_sequences: pd.DataFrame,
                     df_labels: pd.DataFrame,
                     df_splits: pd.DataFrame,
                     sequences_path: Path = SEQUENCES_CSV,
                     labels_path: Path = LABELS_CSV,
                     splits_path: Path = SPLITS_CSV) -> None:
    """
    Save sequences, labels, and splits to CSV files.
    
    Args:
        df_sequences: Sequences DataFrame
        df_labels: Labels DataFrame
        df_splits: Splits DataFrame
        sequences_path: Output path for sequences
        labels_path: Output path for labels
        splits_path: Output path for splits
    """
    logger.info("Saving core tables...")
    
    try:
        # Ensure directories exist
        sequences_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save each table
        df_sequences.to_csv(sequences_path, index=False)
        logger.info(f"Saved sequences: {sequences_path}")
        
        df_labels.to_csv(labels_path, index=False)
        logger.info(f"Saved labels: {labels_path}")
        
        df_splits.to_csv(splits_path, index=False)
        logger.info(f"Saved splits: {splits_path}")
        
        logger.info("All core tables saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving core tables: {e}")
        raise


def load_core_tables(sequences_path: Path = SEQUENCES_CSV,
                     labels_path: Path = LABELS_CSV,
                     splits_path: Path = SPLITS_CSV) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load sequences, labels, and splits from CSV files.
    
    Args:
        sequences_path: Path to sequences CSV
        labels_path: Path to labels CSV
        splits_path: Path to splits CSV
        
    Returns:
        Tuple of (sequences_df, labels_df, splits_df)
        
    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    logger.info("Loading core tables...")
    
    try:
        df_sequences = pd.read_csv(sequences_path)
        logger.info(f"Loaded sequences: {len(df_sequences)} rows")
        
        df_labels = pd.read_csv(labels_path)
        logger.info(f"Loaded labels: {len(df_labels)} rows")
        
        df_splits = pd.read_csv(splits_path)
        logger.info(f"Loaded splits: {len(df_splits)} rows")
        
        return df_sequences, df_labels, df_splits
        
    except Exception as e:
        logger.error(f"Error loading core tables: {e}")
        raise


# ============================================================================
# MAIN FUNCTION FOR TESTING
# ============================================================================

def main():
    """Main function for testing dataset utilities."""
    logger.info("=" * 80)
    logger.info("TESTING DATASET UTILITIES")
    logger.info("=" * 80)
    
    # Example usage
    try:
        # Load FASTA file
        if FASTA_FILE.exists():
            df_sequences = load_fasta(FASTA_FILE)
            logger.info(f"\nLoaded {len(df_sequences)} sequences")
            logger.info(f"First sequence: {df_sequences.iloc[0]['protein_id']}")
        else:
            logger.warning(f"FASTA file not found: {FASTA_FILE}")
            logger.info("Creating example data...")
            df_sequences = pd.DataFrame({
                'protein_id': [f'PROT_{i:04d}' for i in range(100)],
                'description': ['Example protein'] * 100,
                'sequence': ['ACDEFGHIKLMNPQRSTVWY' * 20] * 100,
                'length': [400] * 100,
                'species': ['human'] * 100,
            })
        
        # Validate sequences
        is_valid, errors = validate_sequences(df_sequences)
        if not is_valid:
            logger.warning(f"Validation errors: {errors[:5]}")  # Show first 5
        
        # Create synthetic labels
        df_labels = create_synthetic_labels(df_sequences)
        
        # Merge sequences and labels
        df_merged = df_sequences.merge(df_labels, on='protein_id')
        
        # Create splits
        df_with_splits = make_splits(df_merged)
        
        logger.info("\n" + "=" * 80)
        logger.info("DATASET UTILITIES TEST COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()

