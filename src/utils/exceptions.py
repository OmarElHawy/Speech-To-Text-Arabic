"""Custom exception classes (T025)"""


class SpeechToTextError(Exception):
    """Base exception for Speech to Text project"""
    pass


class AudioError(SpeechToTextError):
    """Exception for audio processing errors"""
    pass


class AudioFileNotFoundError(AudioError):
    """Raised when audio file doesn't exist"""
    pass


class AudioFormatError(AudioError):
    """Raised when audio format is invalid"""
    pass


class AudioProcessingError(AudioError):
    """Raised during audio processing operations"""
    pass


class ModelError(SpeechToTextError):
    """Exception for model-related errors"""
    pass


class ModelNotFoundError(ModelError):
    """Raised when model file doesn't exist"""
    pass


class ModelLoadError(ModelError):
    """Raised when model fails to load"""
    pass


class ModelInferenceError(ModelError):
    """Raised during model inference"""
    pass


class ModelTrainingError(ModelError):
    """Raised during model training"""
    pass


class DataError(SpeechToTextError):
    """Exception for data-related errors"""
    pass


class DatasetNotFoundError(DataError):
    """Raised when dataset directory doesn't exist"""
    pass


class DatasetFormatError(DataError):
    """Raised when dataset format is invalid"""
    pass


class DataLoaderError(DataError):
    """Raised during data loading operations"""
    pass


class BatchProcessingError(SpeechToTextError):
    """Exception for batch processing errors"""
    pass


class ConfigError(SpeechToTextError):
    """Exception for configuration errors"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """Raised when config file doesn't exist"""
    pass


class ConfigParseError(ConfigError):
    """Raised when config parsing fails"""
    pass


class ConfigValueError(ConfigError):
    """Raised when required config value is missing"""
    pass


class GPUError(SpeechToTextError):
    """Exception for GPU-related errors"""
    pass


class GPUNotAvailableError(GPUError):
    """Raised when GPU is required but not available"""
    pass


class GPUMemoryError(GPUError):
    """Raised when GPU memory is insufficient"""
    pass


class EvaluationError(SpeechToTextError):
    """Exception for evaluation errors"""
    pass


class MetricComputationError(EvaluationError):
    """Raised when metric computation fails"""
    pass


class CLIError(SpeechToTextError):
    """Exception for CLI-related errors"""
    pass


class CLIArgumentError(CLIError):
    """Raised when CLI arguments are invalid"""
    pass


class CLIExecutionError(CLIError):
    """Raised when CLI command execution fails"""
    pass


class ValidationError(SpeechToTextError):
    """Exception for validation errors"""
    pass


class TimeoutError(SpeechToTextError):
    """Raised when operation times out"""
    pass
