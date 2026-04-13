"""Base model class (T018)"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
from datetime import datetime
import json

from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseModel(ABC, nn.Module):
    """
    Base class for all ASR models
    
    Provides:
    - Checkpoint save/load functionality
    - Model metadata tracking
    - Device management
    - Config persistence
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[torch.device] = None,
        **config
    ):
        """
        Initialize base model
        
        Args:
            model_name: Name of model (for checkpoints)
            device: torch.device to use
            **config: Additional configuration parameters
        """
        super().__init__()
        
        self.model_name = model_name
        self.device_obj = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.created_at = datetime.now().isoformat()
        self.metadata = {}

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass implementation"""
        pass

    @abstractmethod
    def transcribe(self, audio_data: torch.Tensor) -> str:
        """Transcribe audio to text"""
        pass

    def to_device(self) -> 'BaseModel':
        """Move model to device"""
        self.to(self.device_obj)
        return self

    def save_checkpoint(
        self,
        checkpoint_dir: str = "checkpoints",
        checkpoint_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint with config
        
        Args:
            checkpoint_dir: Directory to save to
            checkpoint_name: Custom checkpoint name (uses timestamp if not provided)
            metadata: Additional metadata to save
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        if checkpoint_name is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_name = f"{self.model_name}_{timestamp}"
        
        checkpoint_file = checkpoint_path / f"{checkpoint_name}.pt"
        config_file = checkpoint_path / f"{checkpoint_name}_config.json"
        
        # Save model state
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'model_name': self.model_name,
            'created_at': self.created_at,
            'saved_at': datetime.now().isoformat(),
        }
        
        # Add optional metadata
        if metadata:
            checkpoint['metadata'] = metadata
            self.metadata.update(metadata)
        
        torch.save(checkpoint, checkpoint_file)
        logger.info(f"Checkpoint saved: {checkpoint_file}")
        
        # Save config separately for reference
        config_data = {
            'model_name': self.model_name,
            'config': self.config,
            'metadata': self.metadata,
            'created_at': self.created_at,
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        logger.info(f"Config saved: {config_file}")
        
        return checkpoint_file

    def load_checkpoint(
        self,
        checkpoint_path: str
    ) -> Dict[str, Any]:
        """
        Load model checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Metadata from checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device_obj)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        self.created_at = checkpoint.get('created_at', self.created_at)
        
        if 'metadata' in checkpoint:
            self.metadata = checkpoint['metadata']
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
        return checkpoint.get('metadata', {})

    def get_model_size(self) -> Dict[str, float]:
        """
        Get model size info
        
        Returns:
            Dictionary with parameter counts and memory estimates
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory (float32 = 4 bytes per parameter)
        total_memory_mb = (total_params * 4) / (1024 * 1024)
        trainable_memory_mb = (trainable_params * 4) / (1024 * 1024)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'total_memory_mb': round(total_memory_mb, 2),
            'trainable_memory_mb': round(trainable_memory_mb, 2),
        }

    def set_trainable(self, trainable: bool = True) -> None:
        """
        Enable/disable training for all parameters
        
        Args:
            trainable: Whether to set trainable
        """
        for param in self.parameters():
            param.requires_grad = trainable
        
        status = "Training" if trainable else "Frozen"
        logger.info(f"{status} mode enabled for {self.model_name}")

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return {
            'model_name': self.model_name,
            'device': str(self.device_obj),
            'config': self.config,
            'metadata': self.metadata,
            'created_at': self.created_at,
        }
