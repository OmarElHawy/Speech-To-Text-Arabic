"""AudioFile data model (T027)"""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path


@dataclass
class AudioFile:
    """
    Data model for audio file metadata
    
    Attributes:
        filename: Path to audio file
        duration: Duration in seconds
        format: Audio format (wav, mp3, flac, etc.)
        sample_rate: Sample rate in Hz
        channels: Number of audio channels
    """
    filename: str
    duration: float
    format: str
    sample_rate: int
    channels: int
    
    def __post_init__(self):
        """Validate attributes after initialization"""
        if self.duration <= 0:
            raise ValueError(f"Duration must be positive, got {self.duration}")
        if self.sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {self.sample_rate}")
        if self.channels <= 0:
            raise ValueError(f"Channels must be positive, got {self.channels}")
        if not self.format:
            raise ValueError("Format cannot be empty")
    
    @classmethod
    def from_path(cls, path: str) -> 'AudioFile':
        """
        Create AudioFile from file path by reading metadata
        
        Args:
            path: Path to audio file
            
        Returns:
            AudioFile instance
        """
        from src.utils.audio import AudioProcessor
        
        processor = AudioProcessor()
        info = processor.get_audio_info(path)
        
        return cls(
            filename=path,
            duration=info['duration'],
            format=info['format'],
            sample_rate=info['sample_rate'],
            channels=info['channels']
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'filename': self.filename,
            'duration': self.duration,
            'format': self.format,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'AudioFile':
        """Create from dictionary"""
        return cls(
            filename=data['filename'],
            duration=data['duration'],
            format=data['format'],
            sample_rate=data['sample_rate'],
            channels=data['channels']
        )
    
    def get_file_size_mb(self) -> float:
        """Get file size in MB"""
        path = Path(self.filename)
        if path.exists():
            return path.stat().st_size / (1024 * 1024)
        return 0.0
    
    def is_valid(self) -> bool:
        """Check if audio file exists and is valid"""
        path = Path(self.filename)
        return path.exists() and path.is_file()
    
    def __str__(self) -> str:
        return (
            f"AudioFile(filename='{self.filename}', "
            f"duration={self.duration:.2f}s, "
            f"format={self.format}, "
            f"sample_rate={self.sample_rate}Hz, "
            f"channels={self.channels})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()