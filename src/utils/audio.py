"""Audio processing utilities (T015)"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Utility class for audio loading, preprocessing, and normalization"""

    # Supported audio formats
    SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    DEFAULT_SR = 16000  # Standard sample rate for speech models

    @staticmethod
    def load_audio(
        audio_path: str, 
        sr: int = DEFAULT_SR, 
        mono: bool = True,
        offset: Optional[float] = None,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Load audio file and resample to specified sample rate
        
        Args:
            audio_path: Path to audio file
            sr: Target sample rate (default: 16000 Hz)
            mono: Convert to mono (default: True)
            offset: Start reading after this time (in seconds)
            duration: Only load this much audio (in seconds)
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        audio_path = Path(audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        if audio_path.suffix.lower() not in AudioProcessor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {audio_path.suffix}. "
                f"Supported formats: {AudioProcessor.SUPPORTED_FORMATS}"
            )
        
        try:
            # Load audio using librosa (handles multiple formats)
            audio, sr_orig = librosa.load(
                str(audio_path),
                sr=sr,
                mono=mono,
                offset=offset,
                duration=duration,
            )
            logger.info(f"Loaded audio: {audio_path.name} ({len(audio) / sr:.2f}s at {sr}Hz)")
            return audio, sr
        except Exception as e:
            raise RuntimeError(f"Failed to load audio file {audio_path}: {str(e)}")

    @staticmethod
    def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        Normalize audio to target loudness level (in dB)
        
        Args:
            audio: Audio data (numpy array)
            target_level: Target level in dB (default: -20.0 dB)
            
        Returns:
            Normalized audio
        """
        # Measure RMS level
        rms = np.sqrt(np.mean(audio**2))
        
        if rms > 0:
            # Calculate gain needed to reach target level
            target_rms = 10 ** (target_level / 20.0)
            gain = target_rms / rms
            audio = audio * gain
            # Clip to prevent clipping distortion
            audio = np.clip(audio, -1.0, 1.0)
            logger.debug(f"Normalized audio: RMS {rms:.4f} -> target {target_level}dB")
        
        return audio

    @staticmethod
    def pad_audio(
        audio: np.ndarray, 
        sr: int,
        duration_sec: float
    ) -> np.ndarray:
        """
        Pad or trim audio to specified duration
        
        Args:
            audio: Audio data
            sr: Sample rate
            duration_sec: Target duration in seconds
            
        Returns:
            Padded or trimmed audio
        """
        target_samples = int(duration_sec * sr)
        current_samples = len(audio)
        
        if current_samples < target_samples:
            # Pad with zeros
            padding = target_samples - current_samples
            audio = np.pad(audio, (0, padding), mode="constant", constant_values=0)
            logger.debug(f"Padded audio from {current_samples} to {target_samples} samples")
        elif current_samples > target_samples:
            # Trim
            audio = audio[:target_samples]
            logger.debug(f"Trimmed audio from {current_samples} to {target_samples} samples")
        
        return audio

    @staticmethod
    def chunk_by_silence(
        audio: np.ndarray,
        sr: int,
        top_db: float = 40.0,
        min_duration: float = 0.5
    ) -> list:
        """
        Chunk audio by silence/pauses
        
        Args:
            audio: Audio data
            sr: Sample rate
            top_db: Threshold for silence detection (in dB)
            min_duration: Minimum chunk duration in seconds
            
        Returns:
            List of audio chunks
        """
        # Detect non-silent intervals
        intervals = librosa.effects.split(audio, top_db=top_db)
        
        min_samples = int(min_duration * sr)
        chunks = []
        
        for interval_start, interval_end in intervals:
            duration = interval_end - interval_start
            if duration >= min_samples:
                chunks.append(audio[interval_start:interval_end])
        
        logger.debug(f"Split audio into {len(chunks)} chunks (min {min_duration}s)")
        return chunks

    @staticmethod
    def chunk_by_duration(
        audio: np.ndarray,
        sr: int,
        chunk_duration: float = 10.0,
        overlap_duration: float = 0.0
    ) -> list:
        """
        Chunk audio into fixed-duration segments with optional overlap
        
        Args:
            audio: Audio data
            sr: Sample rate
            chunk_duration: Duration of each chunk in seconds
            overlap_duration: Overlap between chunks in seconds
            
        Returns:
            List of audio chunks
        """
        chunk_samples = int(chunk_duration * sr)
        overlap_samples = int(overlap_duration * sr)
        stride = chunk_samples - overlap_samples
        
        chunks = []
        for start in range(0, len(audio) - chunk_samples + 1, stride):
            chunks.append(audio[start:start + chunk_samples])
        
        # Handle last chunk if there's remaining audio
        if len(audio) % stride != 0:
            remaining = audio[-(chunk_samples):]
            if len(remaining) >= 0.1 * chunk_samples:  # Only if >10% of chunk size
                chunks.append(remaining)
        
        logger.debug(f"Split audio into {len(chunks)} fixed-duration chunks ({chunk_duration}s each)")
        return chunks


def get_audio_info(audio_path: str) -> dict:
    """Get audio file information without loading entire file"""
    audio_path = Path(audio_path)
    
    try:
        # Quick info fetch with librosa
        info, sr = librosa.load(str(audio_path), sr=None, mono=False, dtype=None)
        duration = librosa.get_duration(path=str(audio_path))
        
        return {
            "path": str(audio_path),
            "duration": duration,
            "sample_rate": sr,
            "channels": 1 if len(info.shape) == 1 else info.shape[0],
            "format": audio_path.suffix.lower(),
        }
    except Exception as e:
        logger.error(f"Failed to get audio info for {audio_path}: {str(e)}")
        return {}
