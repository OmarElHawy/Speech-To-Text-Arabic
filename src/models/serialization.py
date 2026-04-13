"""Model serialization utilities (T030)"""

import json
from typing import Any, Union, List
from pathlib import Path

from src.models.audio_file import AudioFile
from src.models.transcription_result import TranscriptionResult
from src.models.segment import Segment
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelSerializer:
    """
    JSON serialization utilities for data models
    
    Handles:
    - AudioFile serialization
    - TranscriptionResult serialization
    - Segment serialization
    - List serialization
    """
    
    @staticmethod
    def serialize_audio_file(audio_file: AudioFile) -> str:
        """Serialize AudioFile to JSON string"""
        return json.dumps(audio_file.to_dict(), ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize_audio_file(json_str: str) -> AudioFile:
        """Deserialize JSON string to AudioFile"""
        data = json.loads(json_str)
        return AudioFile.from_dict(data)
    
    @staticmethod
    def serialize_transcription_result(result: TranscriptionResult) -> str:
        """Serialize TranscriptionResult to JSON string"""
        return json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize_transcription_result(json_str: str) -> TranscriptionResult:
        """Deserialize JSON string to TranscriptionResult"""
        data = json.loads(json_str)
        return TranscriptionResult.from_dict(data)
    
    @staticmethod
    def serialize_segment(segment: Segment) -> str:
        """Serialize Segment to JSON string"""
        return json.dumps(segment.to_dict(), ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize_segment(json_str: str) -> Segment:
        """Deserialize JSON string to Segment"""
        data = json.loads(json_str)
        return Segment.from_dict(data)
    
    @staticmethod
    def serialize_audio_files(audio_files: List[AudioFile]) -> str:
        """Serialize list of AudioFile objects to JSON string"""
        data = [af.to_dict() for af in audio_files]
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize_audio_files(json_str: str) -> List[AudioFile]:
        """Deserialize JSON string to list of AudioFile objects"""
        data = json.loads(json_str)
        return [AudioFile.from_dict(item) for item in data]
    
    @staticmethod
    def serialize_transcription_results(results: List[TranscriptionResult]) -> str:
        """Serialize list of TranscriptionResult objects to JSON string"""
        data = [tr.to_dict() for tr in results]
        return json.dumps(data, ensure_ascii=False, indent=2)
    
    @staticmethod
    def deserialize_transcription_results(json_str: str) -> List[TranscriptionResult]:
        """Deserialize JSON string to list of TranscriptionResult objects"""
        data = json.loads(json_str)
        return [TranscriptionResult.from_dict(item) for item in data]
    
    @staticmethod
    def save_audio_file(audio_file: AudioFile, file_path: Union[str, Path]) -> None:
        """Save AudioFile to JSON file"""
        json_str = ModelSerializer.serialize_audio_file(audio_file)
        Path(file_path).write_text(json_str, encoding='utf-8')
        logger.info(f"Saved AudioFile to {file_path}")
    
    @staticmethod
    def load_audio_file(file_path: Union[str, Path]) -> AudioFile:
        """Load AudioFile from JSON file"""
        json_str = Path(file_path).read_text(encoding='utf-8')
        return ModelSerializer.deserialize_audio_file(json_str)
    
    @staticmethod
    def save_transcription_result(result: TranscriptionResult, file_path: Union[str, Path]) -> None:
        """Save TranscriptionResult to JSON file"""
        json_str = ModelSerializer.serialize_transcription_result(result)
        Path(file_path).write_text(json_str, encoding='utf-8')
        logger.info(f"Saved TranscriptionResult to {file_path}")
    
    @staticmethod
    def load_transcription_result(file_path: Union[str, Path]) -> TranscriptionResult:
        """Load TranscriptionResult from JSON file"""
        json_str = Path(file_path).read_text(encoding='utf-8')
        return ModelSerializer.deserialize_transcription_result(json_str)
    
    @staticmethod
    def save_audio_files(audio_files: List[AudioFile], file_path: Union[str, Path]) -> None:
        """Save list of AudioFile objects to JSON file"""
        json_str = ModelSerializer.serialize_audio_files(audio_files)
        Path(file_path).write_text(json_str, encoding='utf-8')
        logger.info(f"Saved {len(audio_files)} AudioFile objects to {file_path}")
    
    @staticmethod
    def load_audio_files(file_path: Union[str, Path]) -> List[AudioFile]:
        """Load list of AudioFile objects from JSON file"""
        json_str = Path(file_path).read_text(encoding='utf-8')
        return ModelSerializer.deserialize_audio_files(json_str)
    
    @staticmethod
    def save_transcription_results(results: List[TranscriptionResult], file_path: Union[str, Path]) -> None:
        """Save list of TranscriptionResult objects to JSON file"""
        json_str = ModelSerializer.serialize_transcription_results(results)
        Path(file_path).write_text(json_str, encoding='utf-8')
        logger.info(f"Saved {len(results)} TranscriptionResult objects to {file_path}")
    
    @staticmethod
    def load_transcription_results(file_path: Union[str, Path]) -> List[TranscriptionResult]:
        """Load list of TranscriptionResult objects from JSON file"""
        json_str = Path(file_path).read_text(encoding='utf-8')
        return ModelSerializer.deserialize_transcription_results(json_str)


# Convenience functions for easy import
def save_audio_file(audio_file: AudioFile, file_path: Union[str, Path]) -> None:
    """Save AudioFile to JSON file"""
    ModelSerializer.save_audio_file(audio_file, file_path)


def load_audio_file(file_path: Union[str, Path]) -> AudioFile:
    """Load AudioFile from JSON file"""
    return ModelSerializer.load_audio_file(file_path)


def save_transcription_result(result: TranscriptionResult, file_path: Union[str, Path]) -> None:
    """Save TranscriptionResult to JSON file"""
    ModelSerializer.save_transcription_result(result, file_path)


def load_transcription_result(file_path: Union[str, Path]) -> TranscriptionResult:
    """Load TranscriptionResult from JSON file"""
    return ModelSerializer.load_transcription_result(file_path)


def save_audio_files(audio_files: List[AudioFile], file_path: Union[str, Path]) -> None:
    """Save list of AudioFile objects to JSON file"""
    ModelSerializer.save_audio_files(audio_files, file_path)


def load_audio_files(file_path: Union[str, Path]) -> List[AudioFile]:
    """Load list of AudioFile objects from JSON file"""
    return ModelSerializer.load_audio_files(file_path)


def save_transcription_results(results: List[TranscriptionResult], file_path: Union[str, Path]) -> None:
    """Save list of TranscriptionResult objects to JSON file"""
    ModelSerializer.save_transcription_results(results, file_path)


def load_transcription_results(file_path: Union[str, Path]) -> List[TranscriptionResult]:
    """Load list of TranscriptionResult objects from JSON file"""
    return ModelSerializer.load_transcription_results(file_path)