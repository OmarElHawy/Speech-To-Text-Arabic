"""Data loader service (T019)"""

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

from src.utils.audio import AudioProcessor
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioFileInfo:
    """Metadata about an audio file"""
    path: str
    duration: float
    sample_rate: int
    n_channels: int
    transcription: Optional[str] = None
    split: str = "train"  # train, val, test


class AudioDataset(Dataset):
    """
    PyTorch dataset for audio files
    
    Handles:
    - Audio loading and preprocessing
    - Fixed-length padding/chunking
    - Train/val/test splits
    """

    def __init__(
        self,
        audio_files: List[AudioFileInfo],
        audio_processor: AudioProcessor,
        target_duration: float = 10.0,
        pad: bool = True,
    ):
        """
        Initialize audio dataset
        
        Args:
            audio_files: List of AudioFileInfo objects
            audio_processor: AudioProcessor instance
            target_duration: Target duration in seconds
            pad: Whether to pad/trim to target_duration
        """
        self.audio_files = audio_files
        self.audio_processor = audio_processor
        self.target_duration = target_duration
        self.pad = pad
        
        logger.info(
            f"Created AudioDataset with {len(audio_files)} files, "
            f"target_duration={target_duration}s, pad={pad}"
        )

    def __len__(self) -> int:
        return len(self.audio_files)

    def __getitem__(self, idx: int) -> Dict:
        """Get audio sample"""
        file_info = self.audio_files[idx]
        
        try:
            # Load and process audio
            audio, sr = self.audio_processor.load_audio(file_info.path)
            audio = self.audio_processor.normalize_audio(audio)
            
            if self.pad:
                audio = self.audio_processor.pad_audio(
                    audio,
                    sr,
                    duration_sec=self.target_duration
                )
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)
            
            return {
                'audio': audio_tensor,
                'path': file_info.path,
                'duration': file_info.duration,
                'transcription': file_info.transcription,
                'sample_rate': sr,
            }
        except Exception as e:
            logger.error(f"Error loading {file_info.path}: {str(e)}")
            raise


class DataLoaderService:
    """
    Service for managing audio dataset loading and splitting
    
    Handles:
    - Reading audio file lists
    - Train/val/test splitting
    - Creating PyTorch DataLoaders
    - Batch collation
    """

    def __init__(self, audio_processor: Optional[AudioProcessor] = None):
        """
        Initialize data loader service
        
        Args:
            audio_processor: AudioProcessor instance (creates if not provided)
        """
        self.audio_processor = audio_processor or AudioProcessor()

    def load_file_list(
        self,
        file_list_path: str,
        base_audio_dir: Optional[str] = None
    ) -> List[AudioFileInfo]:
        """
        Load audio file list from JSON
        
        JSON format:
        [
            {
                "path": "path/to/audio.wav",
                "transcription": "text",
                "duration": 10.5
            }
        ]
        
        Args:
            file_list_path: Path to JSON file list
            base_audio_dir: Base directory to prepend to relative paths
            
        Returns:
            List of AudioFileInfo objects
        """
        with open(file_list_path, 'r') as f:
            file_list = json.load(f)
        
        audio_files = []
        for item in file_list:
            path = item['path']
            
            # Prepend base directory if relative path
            if base_audio_dir and not Path(path).is_absolute():
                path = str(Path(base_audio_dir) / path)
            
            # Get audio info
            info = self.audio_processor.get_audio_info(path)
            
            audio_file = AudioFileInfo(
                path=path,
                duration=info['duration'],
                sample_rate=info['sample_rate'],
                n_channels=info['channels'],
                transcription=item.get('transcription'),
            )
            audio_files.append(audio_file)
        
        logger.info(f"Loaded {len(audio_files)} audio files from {file_list_path}")
        return audio_files

    def create_splits(
        self,
        audio_files: List[AudioFileInfo],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[AudioFileInfo], List[AudioFileInfo], List[AudioFileInfo]]:
        """
        Split audio files into train/val/test
        
        Args:
            audio_files: List of AudioFileInfo objects
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        # Calculate split sizes
        total = len(audio_files)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_files, val_files, test_files = random_split(
            audio_files,
            [train_size, val_size, total - train_size - val_size]
        )
        
        # Convert to lists and assign split names
        train_list = [audio_files[i] for i in train_files.indices]
        val_list = [audio_files[i] for i in val_files.indices]
        test_list = [audio_files[i] for i in test_files.indices]
        
        for f in train_list:
            f.split = 'train'
        for f in val_list:
            f.split = 'val'
        for f in test_list:
            f.split = 'test'
        
        logger.info(
            f"Created splits: train={len(train_list)}, "
            f"val={len(val_list)}, test={len(test_list)}"
        )
        
        return train_list, val_list, test_list

    def create_data_loader(
        self,
        audio_files: List[AudioFileInfo],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        target_duration: float = 10.0,
        pad: bool = True,
    ) -> DataLoader:
        """
        Create PyTorch DataLoader
        
        Args:
            audio_files: List of AudioFileInfo objects
            batch_size: Batch size
            shuffle: Whether to shuffle
            num_workers: Number of data loading workers
            target_duration: Target audio duration
            pad: Whether to pad audio
            
        Returns:
            DataLoader instance
        """
        dataset = AudioDataset(
            audio_files=audio_files,
            audio_processor=self.audio_processor,
            target_duration=target_duration,
            pad=pad,
        )
        
        # Custom collate function to handle variable-length sequences
        def collate_fn(batch):
            audios = [item['audio'] for item in batch]
            max_len = max(audio.shape[0] for audio in audios)
            
            # Pad all to max length
            padded = torch.zeros(len(batch), max_len)
            for i, audio in enumerate(audios):
                padded[i, :audio.shape[0]] = audio
            
            return {
                'audio': padded,
                'paths': [item['path'] for item in batch],
                'transcriptions': [item['transcription'] for item in batch],
            }
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )
        
        logger.info(
            f"Created DataLoader with batch_size={batch_size}, "
            f"num_batches={len(dataloader)}"
        )
        
        return dataloader

    def get_dataset_stats(self, audio_files: List[AudioFileInfo]) -> Dict:
        """
        Get statistics about dataset
        
        Args:
            audio_files: List of AudioFileInfo objects
            
        Returns:
            Dictionary with statistics
        """
        durations = [f.duration for f in audio_files]
        total_duration = sum(durations)
        
        return {
            'num_files': len(audio_files),
            'total_duration_sec': total_duration,
            'total_duration_hours': total_duration / 3600,
            'mean_duration_sec': sum(durations) / len(durations),
            'min_duration_sec': min(durations),
            'max_duration_sec': max(durations),
        }
