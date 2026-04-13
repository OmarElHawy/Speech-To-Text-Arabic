"""Segment data model (T029)"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """
    Data model for transcription segment
    
    Attributes:
        start_time: Start time in seconds
        end_time: End time in seconds
        text: Transcribed text for this segment
        confidence_score: Confidence score for this segment (0-1)
    """
    start_time: float
    end_time: float
    text: str
    confidence_score: Optional[float] = None
    
    def __post_init__(self):
        """Validate attributes after initialization"""
        if self.start_time < 0:
            raise ValueError(f"Start time must be non-negative, got {self.start_time}")
        
        if self.end_time <= self.start_time:
            raise ValueError(f"End time must be greater than start time, got {self.end_time} <= {self.start_time}")
        
        if self.confidence_score is not None:
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")
    
    @property
    def duration(self) -> float:
        """Get segment duration in seconds"""
        return self.end_time - self.start_time
    
    def overlaps_with(self, other: 'Segment') -> bool:
        """Check if this segment overlaps with another"""
        return not (self.end_time <= other.start_time or other.end_time <= self.start_time)
    
    def contains_time(self, time: float) -> bool:
        """Check if segment contains the given time"""
        return self.start_time <= time <= self.end_time
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'text': self.text,
            'confidence_score': self.confidence_score,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Segment':
        """Create from dictionary"""
        return cls(
            start_time=data['start_time'],
            end_time=data['end_time'],
            text=data['text'],
            confidence_score=data.get('confidence_score')
        )
    
    def get_word_count(self) -> int:
        """Get number of words in segment"""
        return len(self.text.split())
    
    def get_character_count(self) -> int:
        """Get number of characters in segment"""
        return len(self.text)
    
    def format_time_range(self) -> str:
        """Format time range as string (MM:SS-MM:SS)"""
        def format_time(seconds: float) -> str:
            minutes = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{minutes:02d}:{secs:02d}"
        
        return f"{format_time(self.start_time)}-{format_time(self.end_time)}"
    
    def __str__(self) -> str:
        time_range = self.format_time_range()
        confidence = f", conf={self.confidence_score:.3f}" if self.confidence_score else ""
        return (
            f"Segment({time_range}: '{self.text[:30]}...'{confidence})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()