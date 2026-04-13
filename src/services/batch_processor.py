#!/usr/bin/env python3
"""
Batch Processor Service for Arabic Speech-to-Text System

This service handles batch processing of multiple audio files, providing
parallel processing capabilities and progress tracking for the demo interface.

Author: Speech-to-Text Project Team
Version: 1.0.0
"""

import os
import time
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import asdict
import zipfile
import tempfile

from src.services.demo_service import DemoService
from src.models.transcription_result import TranscriptionResult
from src.utils.config import Config
from src.utils.logging import get_logger
from src.utils.exceptions import AudioProcessingError, BatchProcessingError

class BatchProcessor:
    """
    Service for batch processing multiple audio files.

    This service provides:
    - Parallel processing of audio files
    - Progress tracking and callbacks
    - Result aggregation and ZIP file creation
    - Error handling and recovery
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the batch processor.

        Args:
            max_workers: Maximum number of worker threads (default: CPU count)
        """
        self.logger = get_logger(__name__)
        self.config = Config()
        self.demo_service = DemoService()

        # Threading configuration
        self.max_workers = max_workers or min(4, os.cpu_count() or 1)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="batch-processor"
        )

        # Processing state
        self._is_processing = False
        self._current_progress = 0.0
        self._total_files = 0
        self._processed_files = 0
        self._lock = threading.Lock()

        self.logger.info(f"BatchProcessor initialized with {self.max_workers} workers")

    def process_batch(
        self,
        audio_paths: List[Union[str, Path]],
        model_name: str = "whisper-small",
        language: str = "ar",
        confidence_threshold: float = 0.0,
        device: str = "auto",
        progress_callback: Optional[Callable[[float, str], None]] = None,
        error_callback: Optional[Callable[[str, Exception], None]] = None
    ) -> Dict[str, Any]:
        """
        Process a batch of audio files.

        Args:
            audio_paths: List of paths to audio files
            model_name: Model to use for transcription
            language: Language code
            confidence_threshold: Minimum confidence threshold
            device: Device for inference
            progress_callback: Callback for progress updates (progress, message)
            error_callback: Callback for error handling (filename, error)

        Returns:
            Dict containing:
                - results: List of successful results
                - errors: List of errors
                - summary: Processing summary
                - zip_path: Path to ZIP file with all results

        Raises:
            BatchProcessingError: If batch processing fails
        """
        if self._is_processing:
            raise BatchProcessingError("Batch processing already in progress")

        try:
            self._is_processing = True
            self._total_files = len(audio_paths)
            self._processed_files = 0
            self._current_progress = 0.0

            self.logger.info(f"Starting batch processing of {self._total_files} files")

            # Validate input files
            valid_paths = []
            for path in audio_paths:
                path = Path(path)
                if not path.exists():
                    error_msg = f"File not found: {path}"
                    self.logger.warning(error_msg)
                    if error_callback:
                        error_callback(str(path), FileNotFoundError(error_msg))
                    continue
                valid_paths.append(path)

            if not valid_paths:
                raise BatchProcessingError("No valid audio files found")

            # Update total count
            self._total_files = len(valid_paths)

            # Submit jobs to thread pool
            future_to_path = {}
            for audio_path in valid_paths:
                future = self._executor.submit(
                    self._process_single_file,
                    audio_path,
                    model_name,
                    language,
                    confidence_threshold,
                    device
                )
                future_to_path[future] = audio_path

            # Collect results
            results = []
            errors = []

            for future in concurrent.futures.as_completed(future_to_path):
                audio_path = future_to_path[future]

                try:
                    result = future.result()
                    results.append(result)

                    # Update progress
                    with self._lock:
                        self._processed_files += 1
                        self._current_progress = self._processed_files / self._total_files

                    if progress_callback:
                        progress_callback(
                            self._current_progress,
                            f"Processed {self._processed_files}/{self._total_files}: {audio_path.name}"
                        )

                except Exception as e:
                    error_info = {
                        "filename": audio_path.name,
                        "path": str(audio_path),
                        "error": str(e),
                        "error_type": type(e).__name__
                    }
                    errors.append(error_info)

                    self.logger.error(f"Failed to process {audio_path.name}: {e}")
                    if error_callback:
                        error_callback(audio_path.name, e)

                    # Still update progress for failed files
                    with self._lock:
                        self._processed_files += 1
                        self._current_progress = self._processed_files / self._total_files

                    if progress_callback:
                        progress_callback(
                            self._current_progress,
                            f"Failed {self._processed_files}/{self._total_files}: {audio_path.name}"
                        )

            # Create ZIP file with results
            zip_path = self._create_results_zip(results)

            # Create summary
            summary = {
                "total_files": self._total_files,
                "successful": len(results),
                "failed": len(errors),
                "success_rate": len(results) / self._total_files if self._total_files > 0 else 0,
                "total_processing_time": sum(r.get("processing_time", 0) for r in results),
                "average_processing_time": sum(r.get("processing_time", 0) for r in results) / len(results) if results else 0
            }

            self.logger.info(f"Batch processing completed: {summary}")

            return {
                "results": results,
                "errors": errors,
                "summary": summary,
                "zip_path": zip_path
            }

        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            raise BatchProcessingError(f"Batch processing failed: {str(e)}")

        finally:
            self._is_processing = False

    def _process_single_file(
        self,
        audio_path: Path,
        model_name: str,
        language: str,
        confidence_threshold: float,
        device: str
    ) -> Dict[str, Any]:
        """
        Process a single audio file.

        Args:
            audio_path: Path to the audio file
            model_name: Model name
            language: Language code
            confidence_threshold: Confidence threshold
            device: Device for inference

        Returns:
            Dict with processing result
        """
        start_time = time.time()

        try:
            # Transcribe the file
            result = self.demo_service.transcribe_audio(
                audio_path=audio_path,
                model_name=model_name,
                language=language,
                confidence_threshold=confidence_threshold,
                device=device
            )

            processing_time = time.time() - start_time

            return {
                "filename": audio_path.name,
                "path": str(audio_path),
                "result": result,
                "processing_time": processing_time,
                "success": True
            }

        except Exception as e:
            processing_time = time.time() - start_time
            raise Exception(f"Processing failed after {processing_time:.2f}s: {str(e)}")

    def _create_results_zip(self, results: List[Dict[str, Any]]) -> str:
        """
        Create a ZIP file containing all transcription results.

        Args:
            results: List of successful processing results

        Returns:
            str: Path to the created ZIP file
        """
        # Create temporary ZIP file
        zip_path = tempfile.mktemp(suffix='.zip')

        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for result_data in results:
                    if "result" in result_data and result_data["result"]:
                        result = result_data["result"]
                        filename = result_data["filename"]

                        # Create JSON filename
                        json_filename = f"{Path(filename).stem}_result.json"

                        # Convert result to JSON
                        if hasattr(result, 'to_dict'):
                            result_dict = result.to_dict()
                        else:
                            result_dict = asdict(result)

                        # Add JSON file to ZIP
                        import json
                        json_content = json.dumps(
                            result_dict,
                            indent=2,
                            ensure_ascii=False
                        )
                        zf.writestr(json_filename, json_content)

                # Add a summary file
                summary = {
                    "batch_summary": {
                        "total_files": len(results),
                        "processing_timestamp": time.time(),
                        "model_used": "whisper-small",  # Could be parameterized
                        "language": "ar"
                    },
                    "files": [
                        {
                            "filename": r["filename"],
                            "processing_time": r["processing_time"],
                            "text_length": len(r["result"].text) if r["result"] else 0
                        }
                        for r in results
                    ]
                }

                zf.writestr("batch_summary.json", json.dumps(summary, indent=2))

            self.logger.info(f"Created results ZIP: {zip_path}")
            return zip_path

        except Exception as e:
            self.logger.error(f"Failed to create results ZIP: {e}")
            # Return empty string if ZIP creation fails
            return ""

    def get_progress(self) -> Dict[str, Any]:
        """
        Get current processing progress.

        Returns:
            Dict with progress information
        """
        with self._lock:
            return {
                "is_processing": self._is_processing,
                "progress": self._current_progress,
                "processed_files": self._processed_files,
                "total_files": self._total_files,
                "progress_percentage": self._current_progress * 100
            }

    def cancel_processing(self):
        """Cancel current batch processing."""
        if self._is_processing:
            self.logger.info("Cancelling batch processing")
            self._executor.shutdown(wait=False)
            self._is_processing = False

            # Reinitialize executor for future use
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="batch-processor"
            )

    def __del__(self):
        """Cleanup executor on destruction."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)