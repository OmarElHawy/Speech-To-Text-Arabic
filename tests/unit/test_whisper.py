"""Unit tests for Whisper transcription service (T043)"""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch

from src.models.whisper_base import WhisperBaseModel
from src.models.transcription_result import TranscriptionResult
from src.utils.gpu_config import GPUConfig


class TestWhisperBaseModel:
    """Test WhisperBaseModel functionality"""

    @pytest.fixture
    def gpu_config(self):
        """Mock GPU config"""
        config = Mock(spec=GPUConfig)
        config.get_device.return_value = torch.device('cpu')
        return config

    @pytest.fixture
    def whisper_model(self, gpu_config):
        """Create Whisper model instance"""
        with patch('src.utils.gpu_config.GPUConfig', return_value=gpu_config):
            model = WhisperBaseModel(model_size="tiny")  # Use tiny for faster tests
            return model

    def test_initialization(self, whisper_model):
        """Test model initialization"""
        assert whisper_model.model_name == "whisper-tiny"
        assert whisper_model.model_size == "tiny"
        assert not whisper_model.is_loaded
        assert whisper_model.device.type == "cpu"

    def test_supported_languages(self, whisper_model):
        """Test supported languages"""
        languages = whisper_model.get_supported_languages()
        assert isinstance(languages, list)
        assert "ar" in languages  # Arabic should be supported
        assert "en" in languages  # English should be supported

    def test_model_info_before_loading(self, whisper_model):
        """Test model info before loading"""
        info = whisper_model.get_model_info()
        assert info["loaded"] is False

    @pytest.mark.slow
    def test_model_loading(self, whisper_model):
        """Test model loading (marked as slow)"""
        whisper_model.load_model()
        assert whisper_model.is_loaded

        info = whisper_model.get_model_info()
        assert info["loaded"] is True
        assert "whisper-tiny" in info["model_name"]
        assert info["device"] == "cpu"

    @pytest.mark.slow
    def test_transcribe_audio(self, whisper_model):
        """Test audio transcription"""
        # Load model
        whisper_model.load_model()

        # Create test audio (1 second of silence)
        audio = torch.zeros(16000)

        # Transcribe
        result = whisper_model.transcribe_audio(audio, language="ar")

        # Verify result
        assert isinstance(result, TranscriptionResult)
        assert isinstance(result.text, str)
        assert result.confidence_score is not None
        assert result.language == "ar"
        assert result.model_name == "whisper-tiny"

    def test_memory_usage(self, whisper_model):
        """Test memory usage reporting"""
        usage = whisper_model.get_memory_usage()
        assert "model_loaded" in usage
        assert usage["model_loaded"] is False

    def test_unload_model(self, whisper_model):
        """Test model unloading"""
        whisper_model.load_model()
        assert whisper_model.is_loaded

        whisper_model.unload_model()
        assert not whisper_model.is_loaded

    def test_transcribe_without_loading(self, whisper_model):
        """Test that transcribe fails when model not loaded"""
        audio = torch.zeros(16000)

        with pytest.raises(RuntimeError, match="Model not loaded"):
            whisper_model.transcribe_audio(audio)

    def test_invalid_model_size(self):
        """Test invalid model size"""
        with pytest.raises(ValueError):
            WhisperBaseModel(model_size="invalid_size")


class TestTranscriptionResult:
    """Test TranscriptionResult data model"""

    def test_transcription_result_creation(self):
        """Test creating a TranscriptionResult"""
        result = TranscriptionResult(
            text="Hello world",
            confidence_score=0.95,
            processing_time_ms=1000
        )

        assert result.text == "Hello world"
        assert result.confidence_score == 0.95
        assert result.processing_time_ms == 1000

    def test_transcription_result_validation(self):
        """Test validation of TranscriptionResult fields"""
        # Valid result
        result = TranscriptionResult(text="Test")
        assert result.text == "Test"

        # Invalid confidence score
        with pytest.raises(ValueError):
            TranscriptionResult(text="Test", confidence_score=1.5)

        # Invalid processing time
        with pytest.raises(ValueError):
            TranscriptionResult(text="Test", processing_time_ms=-1)

    def test_transcription_result_to_dict(self):
        """Test converting TranscriptionResult to dict"""
        result = TranscriptionResult(
            text="Hello",
            confidence_score=0.9,
            processing_time_ms=500
        )

        data = result.to_dict()
        assert data["text"] == "Hello"
        assert data["confidence_score"] == 0.9
        assert data["processing_time_ms"] == 500

    def test_transcription_result_from_dict(self):
        """Test creating TranscriptionResult from dict"""
        data = {
            "text": "Hello",
            "confidence_score": 0.9,
            "processing_time_ms": 500
        }

        result = TranscriptionResult.from_dict(data)
        assert result.text == "Hello"
        assert result.confidence_score == 0.9
        assert result.processing_time_ms == 500