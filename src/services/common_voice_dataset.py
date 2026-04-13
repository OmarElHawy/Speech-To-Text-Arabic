"""
Common Voice Arabic Dataset Loader (T072-T074)

Provides dataset loading and preprocessing for Mozilla Common Voice Arabic data.
"""

import pandas as pd
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from torch.utils.data import Dataset
import librosa
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


class CommonVoiceDataset(Dataset):
    """
    PyTorch Dataset for Mozilla Common Voice Arabic data.
    
    Loads audio files and transcriptions from TSV files.
    Handles audio preprocessing (loading, resampling, normalization).
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        processor=None,
        max_duration: float = 30.0,
        sample_rate: int = 16000,
        augment: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the cv-corpus-24.0-2025-12-05/ar directory
            split: Dataset split ('train', 'dev', or 'test')
            processor: Whisper processor for feature extraction
            max_duration: Maximum audio duration in seconds (longer files are truncated)
            sample_rate: Target sample rate (Hz)
            augment: Whether to apply audio augmentation
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.processor = processor
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.augment = augment
        
        # Map split names to TSV files
        split_mapping = {
            "train": "train.tsv",
            "dev": "dev.tsv",
            "test": "test.tsv",
            "validation": "dev.tsv"  # Use dev as validation
        }
        
        tsv_file = self.data_dir / split_mapping.get(split, f"{split}.tsv")
        
        if not tsv_file.exists():
            raise FileNotFoundError(f"Dataset file not found: {tsv_file}")
        
        # Load the TSV file
        logger.info(f"Loading {split} set from {tsv_file}")
        self.df = pd.read_csv(tsv_file, sep='\t')
        
        # Filter out entries with missing required columns
        required_columns = ['path', 'sentence']
        missing = [col for col in required_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Remove rows with missing sentences or paths
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=['path', 'sentence'])
        final_size = len(self.df)
        
        if initial_size > final_size:
            logger.warning(f"Filtered out {initial_size - final_size} entries with missing data")
        
        logger.info(f"Loaded {final_size} samples for {split} split")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary with 'input_features', 'labels', and 'text' keys
        """
        row = self.df.iloc[idx]
        audio_path = self.data_dir / "clips" / row['path']
        text = row['sentence']
        
        # Load audio
        try:
            audio, sr = librosa.load(
                str(audio_path),
                sr=self.sample_rate,
                mono=True
            )
            
            # Truncate if longer than max_duration
            max_samples = int(self.max_duration * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            # Apply augmentation if requested
            if self.augment:
                audio = self._augment_audio(audio)
            
            # Normalize
            audio = audio / (np.max(np.abs(audio)) + 1e-10)
            
        except Exception as e:
            logger.warning(f"Failed to load audio {audio_path}: {e}")
            # Return silent audio of appropriate length
            audio = np.zeros(int(self.sample_rate * 0.5))
        
        # Process with Whisper processor if provided
        if self.processor is not None:
            input_features = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            ).input_features[0]
            
            # Encode text to token IDs
            labels = self.processor.tokenizer(text).input_ids
            
            return {
                "input_features": input_features,
                "labels": torch.tensor(labels, dtype=torch.long),
                "text": text
            }
        else:
            # Return raw audio if no processor provided
            return {
                "audio": torch.FloatTensor(audio),
                "text": text,
                "sample_rate": self.sample_rate
            }
    
    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply audio augmentation (pitch shift, time stretch, noise).
        
        Args:
            audio: Input audio array
            
        Returns:
            Augmented audio array
        """
        # Random pitch shift (-2 to +2 semitones)
        if np.random.random() < 0.5:
            n_steps = np.random.randint(-2, 3)
            audio = librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
        
        # Random time stretch (0.9x to 1.1x)
        if np.random.random() < 0.5:
            rate = np.random.uniform(0.9, 1.1)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        # Random background noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.005, len(audio))
            audio = audio + noise
        
        return audio
    
    @staticmethod
    def create_splits(
        data_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_seed: int = 42
    ) -> Tuple['CommonVoiceDataset', 'CommonVoiceDataset', 'CommonVoiceDataset']:
        """
        Create train/val/test splits from the full dataset.
        
        Args:
            data_dir: Path to the cv-corpus directory
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load full dataset
        logger.info("Creating train/val/test splits")
        full_dataset = CommonVoiceDataset(data_dir, split="train", processor=None)
        
        # Split indices
        np.random.seed(random_seed)
        indices = np.random.permutation(len(full_dataset))
        
        train_size = int(len(full_dataset) * train_ratio)
        val_size = int(len(full_dataset) * val_ratio)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Create subset datasets
        from torch.utils.data import Subset
        
        train_dataset = Subset(full_dataset, train_indices)
        val_dataset = Subset(full_dataset, val_indices)
        test_dataset = Subset(full_dataset, test_indices)
        
        logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
