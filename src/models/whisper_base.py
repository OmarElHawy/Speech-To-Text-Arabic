"""Whisper baseline model implementation (T035)"""

import torch
from typing import List, Optional, Dict, Any
import time

from src.services.transcription_service import TranscriptionService
from src.models.transcription_result import TranscriptionResult
from src.models.segment import Segment
from src.utils.logging import get_logger
from src.utils.gpu_config import GPUConfig

logger = get_logger(__name__)


class WhisperBaseModel(TranscriptionService):
    """
    OpenAI Whisper baseline model for Arabic ASR
    
    Uses whisper-small model from HuggingFace transformers
    """
    
    def __init__(
        self,
        model_size: str = "small",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize Whisper model
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to run on ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache model files
        """
        super().__init__(f"whisper-{model_size}", device)
        
        self.model_size = model_size
        self.cache_dir = cache_dir
        self.model = None
        self.processor = None
        
        # Set device
        if device is None:
            gpu_config = GPUConfig()
            self.device = gpu_config.get_device()
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initialized WhisperBaseModel with size={model_size}, device={self.device}")
    
    def load_model(self) -> None:
        """Load Whisper model and processor"""
        if self.is_loaded:
            logger.warning("Model already loaded")
            return
        
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            logger.info(f"Loading Whisper {self.model_size} model...")
            
            # Load processor
            self.processor = WhisperProcessor.from_pretrained(
                f"openai/whisper-{self.model_size}",
                cache_dir=self.cache_dir
            )
            
            # Load model
            self.model = WhisperForConditionalGeneration.from_pretrained(
                f"openai/whisper-{self.model_size}",
                cache_dir=self.cache_dir
            )
            
            # Move to device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            
            logger.info(f"Successfully loaded Whisper {self.model_size} model on {self.device}")
            
        except ImportError as e:
            raise ImportError(
                "transformers library not found. Install with: pip install transformers"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to load Whisper model: {str(e)}") from e
    
    def unload_model(self) -> None:
        """Unload model from memory"""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            self.is_loaded = False
            
            # Clear GPU cache if using CUDA
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            logger.info("Unloaded Whisper model")
    
    def transcribe_audio(
        self,
        audio_data: Any,
        language: Optional[str] = "ar",
        task: str = "transcribe",
        return_timestamps: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe audio data using Whisper
        
        Args:
            audio_data: Audio tensor (expects shape [samples])
            language: Language code (default: 'ar' for Arabic)
            task: Task type ('transcribe' or 'translate')
            return_timestamps: Whether to return timestamp information
            **kwargs: Additional parameters
            
        Returns:
            TranscriptionResult object
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if not isinstance(audio_data, torch.Tensor):
            audio_data = torch.FloatTensor(audio_data)
        
        # Ensure audio is on CPU for processing
        audio_numpy = audio_data.cpu().numpy()
        
        # Prepare inputs
        inputs = self.processor(
            audio_numpy,
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Set language and task
        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=language,
            task=task
        )
        
        # Generate transcription
        with torch.no_grad():
            generated_ids = self.model.generate(
                inputs["input_features"],
                forced_decoder_ids=forced_decoder_ids,
                max_new_tokens=kwargs.get('max_new_tokens', 445),  # Reduced from 448 to fit within 448 total
                return_timestamps=return_timestamps,
                **kwargs
            )
        
        # Decode results
        if return_timestamps:
            # For now, don't use timestamps as the API doesn't return them as expected
            # TODO: Implement proper timestamp extraction
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            text = transcription
            segments = None
        else:
            transcription = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            text = transcription  # transcription is already a string
            segments = None
        
        # Calculate confidence (using sequence probability if available)
        confidence_score = None
        try:
            # Get log probabilities for confidence estimation
            with torch.no_grad():
                outputs = self.model(**inputs, labels=generated_ids)
                log_probs = outputs.loss  # Negative log likelihood
                confidence_score = float(torch.exp(-log_probs).item())
        except Exception:
            # Fallback: no confidence score
            pass
        
        result = TranscriptionResult(
            text=text,
            confidence_score=confidence_score,
            segments=segments
        )
        
        return result
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        # Whisper supports many languages
        return [
            "ar", "en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh",
            "ko", "hi", "th", "vi", "tr", "pl", "nl", "sv", "da", "no",
            "fi", "cs", "hu", "ro", "bg", "hr", "sl", "sk", "et", "lv",
            "lt", "mt", "ga", "cy", "eu", "is", "fo", "kl"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"loaded": False}
        
        # Get parameter count
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "loaded": True,
            "model_name": f"openai/whisper-{self.model_size}",
            "model_size": self.model_size,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "supported_languages": self.get_supported_languages(),
            "input_sample_rate": 16000,
            "max_input_length": 30.0,  # seconds
        }
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage information"""
        if not self.is_loaded:
            return {"model_loaded": False}
        
        # Calculate memory usage
        total_params = sum(p.numel() for p in self.model.parameters())
        memory_bytes = total_params * 4  # float32 = 4 bytes
        memory_mb = memory_bytes / (1024 * 1024)
        
        return {
            "model_loaded": True,
            "estimated_memory_mb": memory_mb,
            "device": str(self.device),
        }