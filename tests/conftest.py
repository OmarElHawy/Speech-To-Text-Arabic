"""Pytest configuration (T023)"""

import pytest
import torch
from pathlib import Path
from src.utils.gpu_config import GPUConfig
from src.utils.audio import AudioProcessor
from src.utils.logging import setup_simple_logging


@pytest.fixture(scope="session")
def setup_logging_fixture():
    """Setup logging for tests"""
    setup_simple_logging("DEBUG")


@pytest.fixture(scope="session")
def gpu_config():
    """Provide GPU configuration"""
    return GPUConfig()


@pytest.fixture(scope="session")
def device():
    """Provide torch device"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture(scope="session")
def audio_processor():
    """Provide audio processor"""
    return AudioProcessor()


@pytest.fixture(scope="session")
def test_data_dir():
    """Provide test data directory"""
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)
    return test_dir


@pytest.fixture(scope="session")
def test_audio_file(test_data_dir, audio_processor):
    """Create a test audio file"""
    import numpy as np
    import soundfile as sf
    
    test_file = test_data_dir / "test_audio.wav"
    
    if not test_file.exists():
        # Generate 5 seconds of silence (or very quiet noise)
        sr = 16000
        duration = 5
        audio = np.random.randn(sr * duration) * 0.001  # Very quiet
        sf.write(test_file, audio, sr)
    
    return test_file


@pytest.fixture(scope="session")
def test_speech_file(test_data_dir):
    """Create a test speech file with some content"""
    import numpy as np
    import soundfile as sf
    
    test_file = test_data_dir / "test_speech.wav"
    
    if not test_file.exists():
        # Generate 5 seconds of synthetic speech-like content
        sr = 16000
        duration = 5
        t = np.linspace(0, duration, sr * duration)
        
        # Simulate speech with modulation
        frequency = 200 + 100 * np.sin(2 * np.pi * t * 1)  # Varying frequency
        audio = 0.1 * np.sin(2 * np.pi * frequency * t)
        sf.write(test_file, audio, sr)
    
    return test_file


@pytest.fixture
def temp_config_file(tmp_path):
    """Create temporary config file"""
    import yaml
    
    config = {
        'inference': {
            'batch_size': 32,
            'max_length': 480000,
        },
        'training': {
            'learning_rate': 1e-4,
            'epochs': 10,
        },
    }
    
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    
    return config_file


# Pytest options
def pytest_configure(config):
    """Pytest configuration hook"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests as requiring GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


# Custom assertions
class AudioAssertions:
    """Custom assertions for audio operations"""
    
    @staticmethod
    def assert_audio_shape(audio, expected_shape=None):
        """Assert audio tensor shape"""
        assert isinstance(audio, torch.Tensor) or isinstance(audio, np.ndarray)
        if expected_shape:
            assert audio.shape == expected_shape
    
    @staticmethod
    def assert_audio_range(audio, min_val=-1.0, max_val=1.0):
        """Assert audio is in valid range"""
        assert audio.min() >= min_val, f"Audio min {audio.min()} < {min_val}"
        assert audio.max() <= max_val, f"Audio max {audio.max()} > {max_val}"


@pytest.fixture
def audio_assertions():
    """Provide audio assertions"""
    return AudioAssertions()
