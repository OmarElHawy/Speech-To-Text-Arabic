"""Evaluation service with metric computation (T020)"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import numpy as np

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result of a transcription"""
    reference: str  # Ground truth
    hypothesis: str  # Model output
    confidence: Optional[float] = None
    processing_time_sec: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Evaluation metrics"""
    wer: float  # Word Error Rate
    cer: float  # Character Error Rate
    mer: float  # Match Error Rate
    num_references: int
    num_hypotheses: int


class EvaluationService:
    """
    Service for evaluating ASR models
    
    Computes:
    - WER (Word Error Rate)
    - CER (Character Error Rate)
    - MER (Match Error Rate)
    - Confidence statistics
    - Per-sample metrics
    """

    @staticmethod
    def _levenshtein_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein (edit) distance between two strings"""
        if len(s1) < len(s2):
            return EvaluationService._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """
        Compute Word Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Model output text
            
        Returns:
            WER value (0-1, where 0 is perfect)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) > 0 else 0.0
        
        distance = EvaluationService._levenshtein_distance(
            ' '.join(ref_words),
            ' '.join(hyp_words)
        )
        
        return distance / len(ref_words)

    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """
        Compute Character Error Rate
        
        Args:
            reference: Ground truth text
            hypothesis: Model output text
            
        Returns:
            CER value (0-1, where 0 is perfect)
        """
        ref_chars = reference.lower()
        hyp_chars = hypothesis.lower()
        
        if len(ref_chars) == 0:
            return 1.0 if len(hyp_chars) > 0 else 0.0
        
        distance = EvaluationService._levenshtein_distance(ref_chars, hyp_chars)
        
        return distance / len(ref_chars)

    @staticmethod
    def compute_mer(reference: str, hypothesis: str) -> float:
        """
        Compute Match Error Rate (ratio of matching words)
        
        Args:
            reference: Ground truth text
            hypothesis: Model output text
            
        Returns:
            MER value (0-1, where 1 is perfect match)
        """
        ref_words = reference.lower().split()
        hyp_words = hypothesis.lower().split()
        
        if len(ref_words) == 0:
            return 1.0 if len(hyp_words) == 0 else 0.0
        
        matches = sum(1 for i, w in enumerate(hyp_words) 
                     if i < len(ref_words) and w == ref_words[i])
        
        return matches / max(len(ref_words), len(hyp_words))

    @staticmethod
    def evaluate_batch(
        results: List[TranscriptionResult],
        compute_cer: bool = True,
        compute_mer: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate a batch of transcriptions
        
        Args:
            results: List of TranscriptionResult objects
            compute_cer: Whether to compute CER
            compute_mer: Whether to compute MER
            
        Returns:
            EvaluationMetrics object
        """
        wers = []
        cers = []
        mers = []
        
        for result in results:
            wer = EvaluationService.compute_wer(
                result.reference,
                result.hypothesis
            )
            wers.append(wer)
            
            if compute_cer:
                cer = EvaluationService.compute_cer(
                    result.reference,
                    result.hypothesis
                )
                cers.append(cer)
            
            if compute_mer:
                mer = EvaluationService.compute_mer(
                    result.reference,
                    result.hypothesis
                )
                mers.append(mer)
        
        metrics = EvaluationMetrics(
            wer=np.mean(wers),
            cer=np.mean(cers) if cers else 0.0,
            mer=np.mean(mers) if mers else 0.0,
            num_references=len(results),
            num_hypotheses=len(results),
        )
        
        logger.info(
            f"Evaluation: WER={metrics.wer:.4f}, "
            f"CER={metrics.cer:.4f}, MER={metrics.mer:.4f}"
        )
        
        return metrics

    @staticmethod
    def get_per_sample_metrics(
        results: List[TranscriptionResult]
    ) -> List[Dict]:
        """
        Get metrics for each sample
        
        Args:
            results: List of TranscriptionResult objects
            
        Returns:
            List of dictionaries with per-sample metrics
        """
        metrics = []
        
        for i, result in enumerate(results):
            sample_metrics = {
                'sample_id': i,
                'reference': result.reference,
                'hypothesis': result.hypothesis,
                'wer': EvaluationService.compute_wer(
                    result.reference,
                    result.hypothesis
                ),
                'cer': EvaluationService.compute_cer(
                    result.reference,
                    result.hypothesis
                ),
                'mer': EvaluationService.compute_mer(
                    result.reference,
                    result.hypothesis
                ),
            }
            
            if result.confidence is not None:
                sample_metrics['confidence'] = result.confidence
            
            if result.processing_time_sec is not None:
                sample_metrics['processing_time_sec'] = result.processing_time_sec
            
            metrics.append(sample_metrics)
        
        return metrics

    @staticmethod
    def get_confidence_stats(results: List[TranscriptionResult]) -> Dict:
        """
        Get statistics on model confidence scores
        
        Args:
            results: List of TranscriptionResult objects
            
        Returns:
            Dictionary with confidence statistics
        """
        confidences = [r.confidence for r in results if r.confidence is not None]
        
        if not confidences:
            return {}
        
        return {
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences),
            'min_confidence': np.min(confidences),
            'max_confidence': np.max(confidences),
        }

    @staticmethod
    def get_timing_stats(results: List[TranscriptionResult]) -> Dict:
        """
        Get statistics on processing times
        
        Args:
            results: List of TranscriptionResult objects
            
        Returns:
            Dictionary with timing statistics
        """
        times = [r.processing_time_sec for r in results 
                if r.processing_time_sec is not None]
        
        if not times:
            return {}
        
        return {
            'mean_processing_time_sec': np.mean(times),
            'std_processing_time_sec': np.std(times),
            'min_processing_time_sec': np.min(times),
            'max_processing_time_sec': np.max(times),
            'total_processing_time_sec': np.sum(times),
        }
