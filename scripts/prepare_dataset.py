#!/usr/bin/env python3
"""Dataset preparation script (T022)

Prepares Mozilla Common Voice Arabic dataset for training:
- Validates audio files exist
- Extracts transcriptions and metadata
- Generates train/val/test splits
- Creates JSON file lists for DataLoader
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd

from src.utils.audio import AudioProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_common_voice_splits(
    dataset_dir: str,
    language: str = "ar"
) -> Dict[str, pd.DataFrame]:
    """
    Load Mozilla Common Voice splits
    
    Args:
        dataset_dir: Root data directory
        language: Language code (default: ar for Arabic)
        
    Returns:
        Dictionary mapping split names to DataFrames
    """
    lang_dir = Path(dataset_dir) / language
    
    if not lang_dir.exists():
        raise FileNotFoundError(f"Language directory not found: {lang_dir}")
    
    splits = {}
    for split_name in ["train", "dev", "test", "invalidated", "other"]:
        split_file = lang_dir / f"{split_name}.tsv"
        
        if split_file.exists():
            df = pd.read_csv(split_file, sep='\t')
            splits[split_name] = df
            logger.info(f"Loaded {split_name}: {len(df)} samples")
        else:
            logger.warning(f"Split file not found: {split_file}")
    
    return splits


def validate_audio_files(
    splits: Dict[str, pd.DataFrame],
    dataset_dir: str,
    language: str = "ar"
) -> Dict[str, List[int]]:
    """
    Validate that audio files exist
    
    Args:
        splits: Dictionary of DataFrames
        dataset_dir: Root data directory
        language: Language code
        
    Returns:
        Dictionary mapping split names to valid indices
    """
    valid_indices = {}
    lang_dir = Path(dataset_dir) / language
    
    for split_name, df in splits.items():
        valid_rows = []
        
        for idx, (_, row) in enumerate(df.iterrows()):
            audio_file = lang_dir / "clips" / row['path']
            
            if not audio_file.exists():
                logger.warning(f"Missing audio: {audio_file}")
                continue
            
            valid_rows.append(idx)
        
        valid_indices[split_name] = valid_rows
        logger.info(
            f"Valid files in {split_name}: {len(valid_rows)}/{len(df)}"
        )
    
    return valid_indices


def prepare_file_list(
    splits: Dict[str, pd.DataFrame],
    valid_indices: Dict[str, List[int]],
    dataset_dir: str,
    language: str = "ar",
    target_splits: List[str] = None
) -> List[Dict]:
    """
    Prepare file list for training
    
    Args:
        splits: Dictionary of DataFrames
        valid_indices: Valid file indices per split
        dataset_dir: Root data directory
        language: Language code
        target_splits: Which splits to include (default: train/dev/test)
        
    Returns:
        List of file dictionaries
    """
    if target_splits is None:
        target_splits = ["train", "dev", "test"]
    
    lang_dir = Path(dataset_dir) / language
    file_list = []
    audio_processor = AudioProcessor()
    
    for split_name in target_splits:
        if split_name not in splits:
            logger.warning(f"Split {split_name} not found")
            continue
        
        df = splits[split_name]
        indices = valid_indices.get(split_name, [])
        
        for idx in indices:
            row = df.iloc[idx]
            audio_file = lang_dir / "clips" / row['path']
            
            try:
                # Get audio info
                info = audio_processor.get_audio_info(str(audio_file))
                
                file_dict = {
                    'path': str(audio_file),
                    'duration': info['duration'],
                    'transcription': row.get('sentence', ''),
                    'client_id': row.get('client_id', ''),
                    'up_votes': row.get('up_votes', 0),
                    'down_votes': row.get('down_votes', 0),
                    'age': row.get('age', ''),
                    'gender': row.get('gender', ''),
                    'accent': row.get('accent', ''),
                    'locale': row.get('locale', ''),
                    'segment': split_name,
                }
                
                # Filter by quality (upvotes > downvotes)
                if file_dict['up_votes'] >= file_dict['down_votes']:
                    file_list.append(file_dict)
            
            except Exception as e:
                logger.error(f"Error processing {audio_file}: {str(e)}")
                continue
    
    logger.info(f"Prepared {len(file_list)} files for dataset")
    return file_list


def create_splits(
    file_list: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/val/test splits from file list
    
    Args:
        file_list: List of file dictionaries
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        
    Returns:
        Tuple of (train_list, val_list, test_list)
    """
    import random
    
    random.seed(42)
    total = len(file_list)
    
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Shuffle
    shuffled = sorted(file_list, key=lambda x: random.random())
    
    train_list = shuffled[:train_size]
    val_list = shuffled[train_size:train_size + val_size]
    test_list = shuffled[train_size + val_size:]
    
    logger.info(
        f"Splits: train={len(train_list)}, "
        f"val={len(val_list)}, test={len(test_list)}"
    )
    
    return train_list, val_list, test_list


def save_file_lists(
    train_list: List[Dict],
    val_list: List[Dict],
    test_list: List[Dict],
    output_dir: str = "config/datasets"
) -> None:
    """
    Save file lists to JSON
    
    Args:
        train_list: Training file list
        val_list: Validation file list
        test_list: Test file list
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_list in [
        ("train", train_list),
        ("val", val_list),
        ("test", test_list),
    ]:
        output_file = output_path / f"common_voice_ar_{split_name}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_list, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {output_file}: {len(split_list)} files")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare Mozilla Common Voice Arabic dataset"
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="cv-corpus-24.0-2025-12-05",
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="ar",
        help="Language code (default: ar)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="config/datasets",
        help="Output directory for JSON file lists"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio"
    )
    
    args = parser.parse_args()
    
    logger.info(f"Preparing dataset from {args.dataset_dir}")
    
    # Load splits
    splits = load_common_voice_splits(args.dataset_dir, args.language)
    
    # Validate files exist
    valid_indices = validate_audio_files(splits, args.dataset_dir, args.language)
    
    # Prepare file list
    file_list = prepare_file_list(
        splits,
        valid_indices,
        args.dataset_dir,
        args.language
    )
    
    # Create splits
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    train_list, val_list, test_list = create_splits(
        file_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=test_ratio,
    )
    
    # Save
    save_file_lists(train_list, val_list, test_list, args.output_dir)
    
    logger.info("Dataset preparation complete!")


if __name__ == "__main__":
    main()
