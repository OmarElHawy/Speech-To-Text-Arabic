"""AudioProcessor service (T031)"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path

from src.utils.audio import AudioProcessor as AudioUtils, get_audio_info
from src.utils.logging import get_logger
from src.models.audio_file import AudioFile

logger = get_logger(__name__)


class AudioProcessorService:
    """
    Service for audio processing in ASR pipeline
    
    Provides:
    - Audio loading and preprocessing for ASR
    - Resampling to 16kHz for Whisper
    - Normalization and padding
    - Chunking by silence or duration
    - Batch processing capabilities
    """
    
    def __init__(self, target_sample_rate: int = 16000):
        """
        Initialize audio processor service
        
        Args:
            target_sample_rate: Target sample rate for ASR models (default: 16kHz for Whisper)
        """
        self.target_sample_rate = target_sample_rate
        self.audio_utils = AudioUtils()
        
        logger.info(f"Initialized AudioProcessorService with target_sample_rate={target_sample_rate}")
    
    def load_and_preprocess_audio(
        self,
        audio_path: str,
        normalize: bool = True,
        target_duration: Optional[float] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Load and preprocess audio for ASR
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio
            target_duration: Target duration in seconds (pad/truncate if specified)
            
        Returns:
            Tuple of (audio_tensor, sample_rate)
        """
        # Load audio
        audio, sr = self.audio_utils.load_audio(audio_path)
        
        # Resample to target rate if needed
        if sr != self.target_sample_rate:
            audio = self.audio_utils.resample_audio(audio, sr, self.target_sample_rate)
            sr = self.target_sample_rate
        
        # Normalize if requested
        if normalize:
            audio = self.audio_utils.normalize_audio(audio)
        
        # Pad/truncate to target duration if specified
        if target_duration is not None:
            audio = self.audio_utils.pad_audio(audio, sr, duration_sec=target_duration)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        logger.debug(f"Processed audio: {audio_path}, shape={audio_tensor.shape}, sr={sr}")
        
        return audio_tensor, sr
    
    def chunk_by_silence(
        self,
        audio_path: str,
        top_db: float = 40.0,
        min_chunk_duration: float = 1.0,
        max_chunk_duration: float = 30.0
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Chunk audio by silence detection
        
        Args:
            audio_path: Path to audio file
            top_db: Silence threshold in dB
            min_chunk_duration: Minimum chunk duration in seconds
            max_chunk_duration: Maximum chunk duration in seconds
            
        Returns:
            List of (audio_chunk, start_time, end_time) tuples
        """
        # Load and preprocess audio
        audio, sr = self.load_and_preprocess_audio(audio_path, normalize=True)
        audio_np = audio.numpy()
        
        # Get silence chunks
        chunks = self.audio_utils.chunk_by_silence(
            audio_np, sr, top_db=top_db
        )
        
        # Filter by duration and convert to tensors
        filtered_chunks = []
        for chunk_audio, start_time, end_time in chunks:
            duration = end_time - start_time
            if min_chunk_duration <= duration <= max_chunk_duration:
                chunk_tensor = torch.FloatTensor(chunk_audio)
                filtered_chunks.append((chunk_tensor, start_time, end_time))
        
        logger.info(f"Chunked {audio_path} into {len(filtered_chunks)} segments")
        
        return filtered_chunks
    
    def chunk_by_duration(
        self,
        audio_path: str,
        chunk_duration: float = 10.0,
        overlap: float = 0.0
    ) -> List[Tuple[torch.Tensor, float, float]]:
        """
        Chunk audio by fixed duration
        
        Args:
            audio_path: Path to audio file
            chunk_duration: Duration of each chunk in seconds
            overlap: Overlap between chunks in seconds
            
        Returns:
            List of (audio_chunk, start_time, end_time) tuples
        """
        # Load and preprocess audio
        audio, sr = self.load_and_preprocess_audio(audio_path, normalize=True)
        audio_np = audio.numpy()
        
        # Get duration chunks
        chunks = self.audio_utils.chunk_by_duration(
            audio_np, sr, chunk_duration=chunk_duration, overlap=overlap
        )
        
        # Convert to tensors
        tensor_chunks = []
        for chunk_audio, start_time, end_time in chunks:
            chunk_tensor = torch.FloatTensor(chunk_audio)
            tensor_chunks.append((chunk_tensor, start_time, end_time))
        
        logger.info(f"Chunked {audio_path} into {len(tensor_chunks)} fixed-duration segments")
        
        return tensor_chunks
    
    def batch_process_audio_files(
        self,
        audio_files: List[AudioFile],
        batch_size: int = 8,
        target_duration: Optional[float] = None
    ) -> List[Tuple[torch.Tensor, AudioFile]]:
        """
        Batch process multiple audio files
        
        Args:
            audio_files: List of AudioFile objects
            batch_size: Number of files to process at once
            target_duration: Target duration for padding
            
        Returns:
            List of (audio_tensor, audio_file) tuples
        """
        results = []
        
        for i in range(0, len(audio_files), batch_size):
            batch = audio_files[i:i + batch_size]
            
            for audio_file in batch:
                try:
                    audio_tensor, sr = self.load_and_preprocess_audio(
                        audio_file.filename,
                        target_duration=target_duration
                    )
                    results.append((audio_tensor, audio_file))
                    
                except Exception as e:
                    logger.error(f"Failed to process {audio_file.filename}: {str(e)}")
                    continue
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}")
        
        logger.info(f"Successfully processed {len(results)}/{len(audio_files)} audio files")
        
        return results
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        Get comprehensive audio information
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio metadata
        """
        return get_audio_info(audio_path)
    
    def validate_audio_file(self, audio_path: str) -> Tuple[bool, str]:
        """
        Validate audio file for ASR processing
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        path = Path(audio_path)
        
        if not path.exists():
            return False, f"File does not exist: {audio_path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {audio_path}"
        
        try:
            info = self.get_audio_info(audio_path)
            
            # Check duration
            if info['duration'] <= 0:
                return False, f"Invalid duration: {info['duration']}"
            
            # Check sample rate (should be reasonable)
            if not (8000 <= info['sample_rate'] <= 48000):
                return False, f"Unsupported sample rate: {info['sample_rate']}"
            
            # Check channels (mono or stereo)
            if info['channels'] not in [1, 2]:
                return False, f"Unsupported number of channels: {info['channels']}"
            
            return True, ""
            
        except Exception as e:
            return False, f"Failed to read audio file: {str(e)}"
    
    def estimate_processing_time(self, audio_files: List[AudioFile]) -> float:
        """
        Estimate total processing time for audio files
        
        Args:
            audio_files: List of AudioFile objects
            
        Returns:
            Estimated processing time in seconds
        """
        total_duration = sum(af.duration for af in audio_files)
        
        # Rough estimate: 0.1 seconds per second of audio for preprocessing
        # This is a conservative estimate and may vary based on hardware
        estimated_time = total_duration * 0.1
        
        return estimated_time