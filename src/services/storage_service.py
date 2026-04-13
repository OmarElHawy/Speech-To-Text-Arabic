"""Storage service for transcription results (T034)"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv
from datetime import datetime

from src.models.transcription_result import TranscriptionResult
from src.models.audio_file import AudioFile
from src.models.serialization import ModelSerializer
from src.utils.logging import get_logger

logger = get_logger(__name__)


class StorageService:
    """
    Service for storing and loading transcription results
    
    Supports:
    - JSON storage of individual results
    - CSV export for analysis
    - Batch storage operations
    - Metadata tracking
    """
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize storage service
        
        Args:
            results_dir: Directory to store results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized StorageService with results_dir={results_dir}")
    
    def save_transcription_result(
        self,
        result: TranscriptionResult,
        audio_path: str,
        model_name: str,
        output_format: str = "json",
        output_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save transcription result
        
        Args:
            result: TranscriptionResult to save
            audio_path: Original audio file path
            model_name: Name of model used
            output_format: Output format ('json', 'txt', 'csv')
            output_path: Custom output path (optional)
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        if output_path:
            file_path = Path(output_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Create filename from audio path and timestamp
            audio_name = Path(audio_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = f"{audio_name}_{model_name}_{timestamp}"
            
            if output_format == "json":
                file_path = self.results_dir / f"{base_name}.json"
            elif output_format == "txt":
                file_path = self.results_dir / f"{base_name}.txt"
            elif output_format == "csv":
                file_path = self.results_dir / f"{base_name}.csv"
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
        
        if output_format == "json" or (output_path and output_path.endswith('.json')):
            self._save_as_json(result, audio_path, model_name, file_path, metadata)
        elif output_format == "txt" or (output_path and output_path.endswith('.txt')):
            self._save_as_txt(result, file_path)
        elif output_format == "csv" or (output_path and output_path.endswith('.csv')):
            self._save_as_csv([result], audio_path, model_name, file_path, metadata)
        else:
            # Default to JSON
            self._save_as_json(result, audio_path, model_name, file_path, metadata)
        
        logger.info(f"Saved transcription result to {file_path}")
        return file_path
    
    def _save_as_json(
        self,
        result: TranscriptionResult,
        audio_path: str,
        model_name: str,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save as JSON with metadata"""
        data = {
            'audio_path': audio_path,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat(),
            'result': result.to_dict(),
        }
        
        if metadata:
            data['metadata'] = metadata
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _save_as_txt(self, result: TranscriptionResult, file_path: Path) -> None:
        """Save as plain text"""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(result.text)
    
    def _save_as_csv(
        self,
        results: List[TranscriptionResult],
        audio_path: str,
        model_name: str,
        file_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save as CSV"""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'audio_path', 'model_name', 'timestamp', 'text',
                'confidence_score', 'word_error_rate', 'processing_time_ms',
                'word_count', 'character_count'
            ])
            
            timestamp = datetime.now().isoformat()
            
            for result in results:
                writer.writerow([
                    audio_path,
                    model_name,
                    timestamp,
                    result.text,
                    result.confidence_score or '',
                    result.word_error_rate or '',
                    result.processing_time_ms or '',
                    result.get_word_count(),
                    result.get_character_count()
                ])
    
    def load_transcription_result(self, file_path: str) -> TranscriptionResult:
        """
        Load transcription result from file
        
        Args:
            file_path: Path to saved result file
            
        Returns:
            TranscriptionResult object
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Result file not found: {file_path}")
        
        if path.suffix == '.json':
            return self._load_from_json(path)
        elif path.suffix == '.txt':
            return self._load_from_txt(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _load_from_json(self, file_path: Path) -> TranscriptionResult:
        """Load from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return TranscriptionResult.from_dict(data['result'])
    
    def _load_from_txt(self, file_path: Path) -> TranscriptionResult:
        """Load from text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
        
        return TranscriptionResult(text=text)
    
    def save_batch_results(
        self,
        results: List[TranscriptionResult],
        audio_paths: List[str],
        model_name: str,
        output_format: str = "json",
        batch_name: Optional[str] = None
    ) -> Path:
        """
        Save batch transcription results
        
        Args:
            results: List of TranscriptionResult objects
            audio_paths: Corresponding audio file paths
            model_name: Name of model used
            output_format: Output format
            batch_name: Name for batch (uses timestamp if not provided)
            
        Returns:
            Path to saved batch file
        """
        if batch_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            batch_name = f"batch_{model_name}_{timestamp}"
        
        if output_format == "json":
            file_path = self.results_dir / f"{batch_name}.json"
            self._save_batch_as_json(results, audio_paths, model_name, file_path)
        elif output_format == "csv":
            file_path = self.results_dir / f"{batch_name}.csv"
            self._save_batch_as_csv(results, audio_paths, model_name, file_path)
        else:
            raise ValueError(f"Unsupported batch output format: {output_format}")
        
        logger.info(f"Saved batch results ({len(results)} items) to {file_path}")
        return file_path
    
    def _save_batch_as_json(
        self,
        results: List[TranscriptionResult],
        audio_paths: List[str],
        model_name: str,
        file_path: Path
    ) -> None:
        """Save batch as JSON"""
        batch_data = {
            'batch_info': {
                'model_name': model_name,
                'timestamp': datetime.now().isoformat(),
                'num_results': len(results)
            },
            'results': []
        }
        
        for result, audio_path in zip(results, audio_paths):
            result_data = {
                'audio_path': audio_path,
                'result': result.to_dict()
            }
            batch_data['results'].append(result_data)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(batch_data, f, ensure_ascii=False, indent=2)
    
    def _save_batch_as_csv(
        self,
        results: List[TranscriptionResult],
        audio_paths: List[str],
        model_name: str,
        file_path: Path
    ) -> None:
        """Save batch as CSV"""
        with open(file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'audio_path', 'model_name', 'timestamp', 'text',
                'confidence_score', 'word_error_rate', 'processing_time_ms',
                'word_count', 'character_count'
            ])
            
            timestamp = datetime.now().isoformat()
            
            for result, audio_path in zip(results, audio_paths):
                writer.writerow([
                    audio_path,
                    model_name,
                    timestamp,
                    result.text,
                    result.confidence_score or '',
                    result.word_error_rate or '',
                    result.processing_time_ms or '',
                    result.get_word_count(),
                    result.get_character_count()
                ])
    
    def load_batch_results(self, file_path: str) -> List[TranscriptionResult]:
        """
        Load batch transcription results
        
        Args:
            file_path: Path to batch result file
            
        Returns:
            List of TranscriptionResult objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Batch result file not found: {file_path}")
        
        if path.suffix == '.json':
            return self._load_batch_from_json(path)
        else:
            raise ValueError(f"Unsupported batch file format: {path.suffix}")
    
    def _load_batch_from_json(self, file_path: Path) -> List[TranscriptionResult]:
        """Load batch from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = []
        for item in data['results']:
            result = TranscriptionResult.from_dict(item['result'])
            results.append(result)
        
        return results
    
    def list_saved_results(self) -> List[Dict[str, Any]]:
        """
        List all saved result files
        
        Returns:
            List of file information dictionaries
        """
        result_files = []
        
        for file_path in self.results_dir.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.json', '.txt', '.csv']:
                stat = file_path.stat()
                result_files.append({
                    'path': str(file_path),
                    'name': file_path.name,
                    'size_bytes': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'format': file_path.suffix[1:]  # Remove leading dot
                })
        
        return sorted(result_files, key=lambda x: x['modified'], reverse=True)