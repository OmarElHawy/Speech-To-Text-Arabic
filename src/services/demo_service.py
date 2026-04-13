#!/usr/bin/env python3
"""
Demo Service for Arabic Speech-to-Text System

This service orchestrates the demo interface operations, handling transcription
requests, batch processing, and result formatting for the web interface.

Author: Speech-to-Text Project Team
Version: 1.0.0
"""

import os
import time
import hashlib
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict

from src.services.transcription_pipeline import TranscriptionPipeline
from src.services.audio_processor import AudioProcessorService
from src.models.whisper_base import WhisperBaseModel
from src.models.transcription_result import TranscriptionResult
from src.models.audio_file import AudioFile
from src.utils.config import Config
from src.utils.logging import get_logger
from src.utils.exceptions import AudioProcessingError, ModelLoadError

class DemoService:
    """
    Service for handling demo interface operations.

    This service provides high-level methods for the web interface to:
    - Transcribe single audio files
    - Handle batch processing requests
    - Format results for display
    - Cache results to avoid recomputation
    """

    def __init__(self):
        """Initialize the demo service."""
        self.logger = get_logger(__name__)
        self.config = Config()

        # Initialize core services
        self.audio_processor = AudioProcessorService()

        # Cache for results (simple in-memory cache)
        self._result_cache: Dict[str, TranscriptionResult] = {}
        self._cache_max_size = self.config.get("demo.cache_max_size", 100)

        # Cache for loaded models
        self._loaded_models: Dict[str, TranscriptionPipeline] = {}

        self.logger.info("DemoService initialized")

    def _resolve_device(self, device: str = "auto") -> str:
        """
        Resolve device string to actual device (cpu or cuda).

        Args:
            device: Device name ('auto', 'cpu', or 'cuda')

        Returns:
            str: Actual device name ('cpu' or 'cuda')
        """
        if device == "auto":
            if torch.cuda.is_available():
                self.logger.info("Auto-detected CUDA device available")
                return "cuda"
            else:
                self.logger.info("Auto-detected CPU device (CUDA not available)")
                return "cpu"
        return device

    def _get_audio_hash(self, audio_path: Union[str, Path]) -> str:
        """
        Generate a hash for audio file content to enable caching.

        Args:
            audio_path: Path to the audio file

        Returns:
            str: SHA256 hash of the audio file
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Use file size and modification time for quick hashing
        stat = audio_path.stat()
        hash_input = f"{audio_path}:{stat.st_size}:{stat.st_mtime}"

        return hashlib.sha256(hash_input.encode()).hexdigest()

    def _get_cache_key(
        self,
        audio_hash: str,
        model_name: str,
        language: str,
        confidence_threshold: float = 0.0,
        device: str = "auto"
    ) -> str:
        """
        Generate a unique cache key for transcription results.

        Args:
            audio_hash: Hash of the audio file
            model_name: Name of the model used
            language: Language used for transcription
            confidence_threshold: Confidence threshold applied
            device: Device used for inference

        Returns:
            str: Unique cache key combining all parameters
        """
        # Combine all parameters into a unique key
        cache_input = f"{audio_hash}:{model_name}:{language}:{confidence_threshold}:{device}"
        return hashlib.sha256(cache_input.encode()).hexdigest()

    def _get_pipeline(self, model_name: str, device: str = "auto") -> TranscriptionPipeline:
        """
        Get or create a transcription pipeline for the specified model.

        Args:
            model_name: Name of the model (e.g., 'whisper-small')
            device: Device to use

        Returns:
            TranscriptionPipeline: Configured pipeline
        """
        cache_key = f"{model_name}:{device}"

        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]

        # Create new pipeline
        if model_name.startswith("whisper-"):
            model_size = model_name.split("-")[1]
            transcription_service = WhisperBaseModel(model_size=model_size, device=device)
            transcription_service.load_model()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        pipeline = TranscriptionPipeline(
            transcription_service=transcription_service,
            audio_processor=self.audio_processor
        )

        # Cache the pipeline
        self._loaded_models[cache_key] = pipeline

        return pipeline

    def transcribe_audio(
        self,
        audio_path: Union[str, Path],
        model_name: str = "whisper-small",
        language: str = "ar",
        confidence_threshold: float = 0.0,
        device: str = "auto",
        use_cache: bool = True
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file.

        Args:
            audio_path: Path to the audio file
            model_name: Name of the model to use
            language: Language code for transcription
            confidence_threshold: Minimum confidence for segments
            device: Device to use for inference
            use_cache: Whether to use cached results

        Returns:
            TranscriptionResult: Transcription results

        Raises:
            AudioProcessingError: If audio processing fails
            ModelLoadError: If model loading fails
        """
        audio_path = Path(audio_path)
        self.logger.info(f"Transcribing audio: {audio_path.name}")

        try:
            # Validate audio file
            if not audio_path.exists():
                raise AudioProcessingError(f"Audio file not found: {audio_path}")

            # Get audio information
            audio_info = self.audio_processor.get_audio_info(str(audio_path))
            if not audio_info or audio_info.get('duration', 0) <= 0:
                raise AudioProcessingError("Invalid audio file: unable to read or zero/negative duration")

            self.logger.debug(f"Audio info: {audio_info}")

            # Resolve device (convert "auto" to actual device)
            device = self._resolve_device(device)

            # Check cache
            if use_cache:
                audio_hash = self._get_audio_hash(audio_path)
                cache_key = self._get_cache_key(
                    audio_hash, model_name, language,
                    confidence_threshold=confidence_threshold,
                    device=device
                )

                if cache_key in self._result_cache:
                    self.logger.info("Using cached transcription result")
                    return self._result_cache[cache_key]

            # Create AudioFile object
            audio_file = AudioFile(
                filename=str(audio_path),
                duration=audio_info.get('duration', 0),
                format=audio_info.get('format', '.wav').lstrip('.'),
                sample_rate=audio_info.get('sample_rate', 16000),
                channels=audio_info.get('channels', 1)
            )

            # Get the appropriate pipeline
            pipeline = self._get_pipeline(model_name, device)

            # Perform transcription
            start_time = time.time()
            result = pipeline.transcribe_file(
                audio_path=str(audio_path),
                language=language,
                generate_segments=True
            )
            processing_time = time.time() - start_time

            # Update result with processing time (in milliseconds)
            result.processing_time_ms = processing_time * 1000

            # Filter segments by confidence if threshold > 0
            if confidence_threshold > 0 and result.segments:
                result.segments = [
                    segment for segment in result.segments
                    if segment.confidence_score and segment.confidence_score >= confidence_threshold
                ]

            # Cache result
            if use_cache:
                if len(self._result_cache) >= self._cache_max_size:
                    # Simple LRU: remove oldest entry
                    oldest_key = next(iter(self._result_cache))
                    del self._result_cache[oldest_key]

                self._result_cache[cache_key] = result

            self.logger.info(f"Transcription completed in {processing_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}")
            raise

    def get_supported_models(self) -> List[str]:
        """
        Get list of supported model names.

        Returns:
            List[str]: Available model names
        """
        return [
            "whisper-small",
            "whisper-base",
            "whisper-medium",
            "whisper-large-v2"
        ]

    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get dictionary of supported languages.

        Returns:
            Dict[str, str]: Language code -> display name mapping
        """
        return {
            "ar": "العربية (Arabic)",
            "en": "English",
            "fr": "Français (French)",
            "es": "Español (Spanish)"
        }

    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information for display in the demo interface.

        Returns:
            Dict[str, Any]: System information
        """
        try:
            import torch
            import gradio as gr
            from utils.gpu_config import GPUConfig

            gpu_config = GPUConfig()
            gpu_info = gpu_config.get_gpu_info()

            return {
                "version": "1.0.0",
                "pytorch_version": torch.__version__,
                "gradio_version": gr.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_info": gpu_info,
                "supported_models": self.get_supported_models(),
                "supported_languages": list(self.get_supported_languages().keys()),
                "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
                "max_file_size_mb": 500,
                "max_duration_hours": 1
            }
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {
                "error": f"Failed to get system info: {str(e)}"
            }

    def clear_cache(self):
        """Clear the result cache."""
        self._result_cache.clear()
        self.logger.info("Result cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict[str, Any]: Cache statistics
        """
        return {
            "cache_size": len(self._result_cache),
            "max_cache_size": self._cache_max_size,
            "cache_hit_ratio": 0.0  # Would need to track hits/misses for this
        }