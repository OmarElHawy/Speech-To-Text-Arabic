"""Transcription pipeline service (T033)"""

from typing import List, Optional, Dict, Any
import time

from src.services.transcription_service import TranscriptionService
from src.services.audio_processor import AudioProcessorService
from src.models.transcription_result import TranscriptionResult
from src.models.segment import Segment
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TranscriptionPipeline:
    """
    End-to-end transcription pipeline
    
    Orchestrates:
    - Audio preprocessing
    - Model inference
    - Result post-processing
    - Segment generation
    """
    
    def __init__(
        self,
        transcription_service: TranscriptionService,
        audio_processor: Optional[AudioProcessorService] = None
    ):
        """
        Initialize transcription pipeline
        
        Args:
            transcription_service: Service for transcription
            audio_processor: Service for audio processing (created if not provided)
        """
        self.transcription_service = transcription_service
        self.audio_processor = audio_processor or AudioProcessorService()
        
        logger.info("Initialized TranscriptionPipeline")
    
    def transcribe_file(
        self,
        audio_path: str,
        language: Optional[str] = None,
        generate_segments: bool = True,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file end-to-end
        
        Args:
            audio_path: Path to audio file
            language: Language code
            generate_segments: Whether to generate segment information
            **kwargs: Additional parameters for transcription service
            
        Returns:
            TranscriptionResult with full text and segments
        """
        start_time = time.time()
        
        # Validate audio file
        is_valid, error_msg = self.audio_processor.validate_audio_file(audio_path)
        if not is_valid:
            raise ValueError(f"Invalid audio file {audio_path}: {error_msg}")
        
        # Get audio info
        audio_info = self.audio_processor.get_audio_info(audio_path)
        
        logger.info(
            f"Starting transcription of {audio_path} "
            f"({audio_info['duration']:.2f}s, {audio_info['format']})"
        )
        
        # For short files, transcribe directly
        if audio_info['duration'] <= 30.0:  # Whisper can handle up to 30s
            result = self.transcription_service.transcribe_file(
                audio_path, language=language, **kwargs
            )
            
            # Generate segments if requested
            if generate_segments and not result.segments:
                result.segments = self._generate_segments_from_result(
                    result, audio_info['duration']
                )
        
        # For long files, chunk and transcribe in segments
        else:
            result = self._transcribe_long_audio(
                audio_path, language=language, generate_segments=generate_segments, **kwargs
            )
        
        # Add processing time
        total_time = time.time() - start_time
        result.processing_time_ms = int(total_time * 1000)
        
        logger.info(
            f"Completed transcription in {result.processing_time_ms}ms, "
            f"text length: {len(result.text)} chars"
        )
        
        return result
    
    def _transcribe_long_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        generate_segments: bool = True,
        chunk_duration: float = 25.0,  # Leave buffer for Whisper
        overlap: float = 2.0,
        **kwargs
    ) -> TranscriptionResult:
        """
        Transcribe long audio by chunking
        
        Args:
            audio_path: Path to audio file
            language: Language code
            generate_segments: Whether to generate segments
            chunk_duration: Duration of each chunk
            overlap: Overlap between chunks
            **kwargs: Additional parameters
            
        Returns:
            Combined TranscriptionResult
        """
        # Chunk audio
        chunks = self.audio_processor.chunk_by_duration(
            audio_path,
            chunk_duration=chunk_duration,
            overlap=overlap
        )
        
        logger.info(f"Processing {len(chunks)} chunks for long audio")
        
        # Transcribe each chunk
        chunk_results = []
        all_segments = []
        total_confidence = 0.0
        chunk_count = 0
        
        for i, (chunk_audio, start_time, end_time) in enumerate(chunks):
            try:
                # Transcribe chunk
                chunk_result = self.transcription_service.transcribe_audio(
                    chunk_audio, language=language, **kwargs
                )
                
                if chunk_result.confidence_score is not None:
                    total_confidence += chunk_result.confidence_score
                    chunk_count += 1
                
                chunk_results.append((chunk_result, start_time, end_time))
                
                # Adjust segment timestamps
                if chunk_result.segments:
                    for segment in chunk_result.segments:
                        segment.start_time += start_time
                        segment.end_time += start_time
                        all_segments.append(segment)
                
                logger.debug(f"Processed chunk {i+1}/{len(chunks)}: {start_time:.1f}s-{end_time:.1f}s")
                
            except Exception as e:
                logger.error(f"Failed to transcribe chunk {i+1}: {str(e)}")
                continue
        
        # Combine results
        combined_text = " ".join([result.text for result, _, _ in chunk_results])
        
        # Calculate average confidence
        avg_confidence = total_confidence / chunk_count if chunk_count > 0 else None
        
        # Create combined result
        result = TranscriptionResult(
            text=combined_text.strip(),
            confidence_score=avg_confidence,
            segments=all_segments if generate_segments else None
        )
        
        return result
    
    def _generate_segments_from_result(
        self,
        result: TranscriptionResult,
        total_duration: float
    ) -> List[Segment]:
        """
        Generate segments from transcription result
        
        Args:
            result: TranscriptionResult
            total_duration: Total audio duration
            
        Returns:
            List of Segment objects
        """
        # Simple segmentation: split by sentences and estimate timing
        import re
        
        sentences = re.split(r'[.!?]+', result.text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return []
        
        # Estimate segment duration (simple equal division)
        segment_duration = total_duration / len(sentences)
        
        segments = []
        current_time = 0.0
        
        for sentence in sentences:
            if not sentence:
                continue
                
            segment = Segment(
                start_time=current_time,
                end_time=current_time + segment_duration,
                text=sentence,
                confidence_score=result.confidence_score
            )
            segments.append(segment)
            current_time += segment_duration
        
        return segments
    
    def transcribe_batch(
        self,
        audio_paths: List[str],
        language: Optional[str] = None,
        **kwargs
    ) -> List[TranscriptionResult]:
        """
        Transcribe multiple audio files
        
        Args:
            audio_paths: List of audio file paths
            language: Language code
            **kwargs: Additional parameters
            
        Returns:
            List of TranscriptionResult objects
        """
        results = []
        
        for audio_path in audio_paths:
            try:
                result = self.transcribe_file(audio_path, language=language, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to transcribe {audio_path}: {str(e)}")
                # Return empty result for failed transcriptions
                error_result = TranscriptionResult(
                    text="",
                    confidence_score=0.0,
                    processing_time_ms=0
                )
                results.append(error_result)
        
        logger.info(f"Batch transcription complete: {len(results)} files processed")
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline
        
        Returns:
            Dictionary with pipeline information
        """
        return {
            'transcription_service': str(self.transcription_service),
            'audio_processor': str(self.audio_processor),
            'supported_languages': self.transcription_service.get_supported_languages(),
            'model_info': self.transcription_service.get_model_info(),
        }