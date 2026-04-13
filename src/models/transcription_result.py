"""TranscriptionResult data model (T028)"""

from dataclasses import dataclass
from typing import Optional, List
from src.models.segment import Segment


@dataclass
class TranscriptionResult:
    """
    Data model for transcription results
    
    Attributes:
        text: Full transcribed text
        confidence_score: Overall confidence score (0-1)
        word_error_rate: Word Error Rate if reference available
        processing_time_ms: Time taken to process in milliseconds
        segments: List of transcription segments (optional)
    """
    text: str
    confidence_score: Optional[float] = None
    word_error_rate: Optional[float] = None
    processing_time_ms: Optional[float] = None
    segments: Optional[List[Segment]] = None
    
    def __post_init__(self):
        """Validate attributes after initialization"""
        if self.confidence_score is not None:
            if not (0.0 <= self.confidence_score <= 1.0):
                raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")
        
        if self.word_error_rate is not None:
            if self.word_error_rate < 0.0:
                raise ValueError(f"Word error rate must be non-negative, got {self.word_error_rate}")
        
        if self.processing_time_ms is not None:
            if self.processing_time_ms < 0:
                raise ValueError(f"Processing time must be non-negative, got {self.processing_time_ms}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        result = {
            'text': self.text,
            'confidence_score': self.confidence_score,
            'word_error_rate': self.word_error_rate,
            'processing_time_ms': self.processing_time_ms,
        }
        
        if self.segments:
            result['segments'] = [segment.to_dict() for segment in self.segments]
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TranscriptionResult':
        """Create from dictionary"""
        segments = None
        if 'segments' in data and data['segments']:
            from src.models.segment import Segment
            segments = [Segment.from_dict(seg) for seg in data['segments']]
        
        return cls(
            text=data['text'],
            confidence_score=data.get('confidence_score'),
            word_error_rate=data.get('word_error_rate'),
            processing_time_ms=data.get('processing_time_ms'),
            segments=segments
        )
    
    def get_word_count(self) -> int:
        """Get number of words in transcription"""
        return len(self.text.split())
    
    def get_character_count(self) -> int:
        """Get number of characters in transcription"""
        return len(self.text)
    
    def has_segments(self) -> bool:
        """Check if result has segment information"""
        return self.segments is not None and len(self.segments) > 0
    
    def get_segment_count(self) -> int:
        """Get number of segments"""
        return len(self.segments) if self.segments else 0
    
    def get_average_segment_confidence(self) -> Optional[float]:
        """Get average confidence across segments"""
        if not self.has_segments():
            return None
        
        confidences = [seg.confidence_score for seg in self.segments 
                      if seg.confidence_score is not None]
        
        return sum(confidences) / len(confidences) if confidences else None
    
    def __str__(self) -> str:
        segments_info = f", segments={len(self.segments)}" if self.segments else ""
        return (
            f"TranscriptionResult(text='{self.text[:50]}...', "
            f"confidence={self.confidence_score:.3f}, "
            f"wer={self.word_error_rate:.3f}, "
            f"time={self.processing_time_ms}ms{segments_info})"
        )
    
    def __repr__(self) -> str:
        return self.__str__()