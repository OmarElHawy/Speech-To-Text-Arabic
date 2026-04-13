"""GPU and CUDA configuration utilities (T014)"""

import torch
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class GPUConfig:
    """Helper class for GPU configuration and device management"""

    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device (CUDA > CPU)"""
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            return torch.device("cuda")
        else:
            logger.info("GPU not available, using CPU")
            return torch.device("cpu")

    @staticmethod
    def get_cuda_info() -> Dict[str, Any]:
        """Get detailed CUDA information"""
        info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
        
        if torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
            info["device_capability"] = torch.cuda.get_device_capability(0)
            info["total_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            try:
                info["allocated_memory_gb"] = torch.cuda.memory_allocated(0) / (1024**3)
                info["reserved_memory_gb"] = torch.cuda.memory_reserved(0) / (1024**3)
            except:
                pass
        
        return info

    @staticmethod
    def clear_cache():
        """Clear CUDA cache if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

    @staticmethod
    def set_seed(seed: int = 42):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Note: Also set numpy/random seeds if needed
        logger.info(f"Random seed set to {seed}")
