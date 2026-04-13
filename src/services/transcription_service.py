"""TranscriptionService base class (T032)"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time

from src.models.transcription_result import TranscriptionResult
from src.models.segment import Segment
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptionService(ABC):
    """
    Base class for all transcription services
    
    Defines the interface that all ASR models must implement
    """
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize transcription service
        
        Args:
            model_name: Name of the model
            device: Device to run model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        self.device = device
        self.is_loaded = False
        
        logger.info(f"Initialized {self.__class__.__name__} with model {model_name}")
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model into memory"""
        pass
    
    @abstractmethod
    def unload_model(self) -> None:
        """Unload the model from memory"""
        pass
    
    @abstractmethod
    def transcribe_audio(
        self,
        audio_data: Any,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data to text
        
        Args:
            audio_data: Audio data (format depends on model)
            language: Language code (optional)
            **kwargs: Additional model-specific parameters
            
        Returns:
            TranscriptionResult object
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and capabilities"""
        pass
    
    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio file
        
        Args:
            audio_path: Path to audio file
            language: Language code (optional)
            **kwargs: Additional parameters
            
        Returns:
            TranscriptionResult object
        """
        from src.services.audio_processor import AudioProcessorService
        
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} is not loaded")
        
        # Load and preprocess audio
        processor = AudioProcessorService()
        audio_tensor, sr = processor.load_and_preprocess_audio(audio_path)
        
        # Transcribe
        start_time = time.time()
        result = self.transcribe_audio(audio_tensor, language=language, **kwargs)
        end_time = time.time()
        
        # Add processing time
        result.processing_time_ms = int((end_time - start_time) * 1000)
        
        logger.info(
            f"Transcribed {audio_path} in {result.processing_time_ms}ms, "
            f"WER: {result.word_error_rate:.3f}" if result.word_error_rate else "no WER"
        )
        
        return result
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of paths to audio files
            language: Language code (optional)
            **kwargs: Additional parameters
            
        Returns:
            List of TranscriptionResult objects
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model {self.model_name} is not loaded")
        
        results = []
        total_start_time = time.time()
        
        for i, audio_path in enumerate(audio_paths):
            try:
                result = self.transcribe_file(audio_path, language=language, **kwargs)
                results.append(result)
                
                logger.info(f"Processed {i+1}/{len(audio_paths)}: {audio_path}")
                
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
                # Create error result
                error_result = TranscriptionResult(
                    text="",
                    confidence_score=0.0,
                    processing_time_ms=0
                )
                results.append(error_result)
        
        total_time = time.time() - total_start_time
        logger.info(
            f"Batch transcription complete: {len(results)}/{len(audio_paths)} "
            f"successful in {total_time:.2f}s"
        )
        
        return results
    
    def validate_model(self) -> bool:
        """
        Validate that the model is properly loaded and functional
        
        Returns:
            True if model is valid and ready
        """
        if not self.is_loaded:
            return False
        
        try:
            # Try to get model info
            info = self.get_model_info()
            return info is not None
        except Exception:
            return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get model memory usage information
        
        Returns:
            Dictionary with memory usage stats
        """
        # Default implementation - subclasses can override for more detailed info
        return {
            'model_loaded': self.is_loaded,
            'estimated_memory_mb': 0.0  # Override in subclasses
        }
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name}, loaded={self.is_loaded})"
    
    def __repr__(self) -> str:
        return self.__str__()