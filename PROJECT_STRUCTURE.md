# Speech-to-Text Project - Complete File Structure Guide

## Overview
This is a **Deep Learning Based Arabic Audio Understanding and Retrieval System** using OpenAI Whisper for Arabic speech-to-text transcription with fine-tuning capabilities.

---

## 📁 Project Directory Structure

```
Speech to Text Project/
├── src/                           # Main source code
│   ├── cli/                       # Command-line interface
│   ├── models/                    # Data models and ML models
│   ├── services/                  # Core business logic services
│   └── utils/                     # Utility functions
├── notebooks/                     # Jupyter notebooks for training
├── scripts/                       # Standalone scripts
├── tests/                         # Unit and integration tests
├── demo/                          # Web interface (Gradio)
├── docs/                          # Documentation
├── config/                        # Configuration files
├── specs/                         # Project specifications
├── README.md                      # Main documentation
├── requirements.txt               # Python dependencies
└── speech-to-text-project.zip    # Zipped project (for GitHub)
```

---

## 📄 Root Level Files

### `README.md`
- **Purpose**: Main project documentation
- **Contains**: 
  - Project overview and features
  - Installation instructions
  - Quick start guide
  - CLI usage examples
  - Demo interface features
  - Architecture overview
  - Model comparison
  - Advanced usage

### `requirements.txt`
- **Purpose**: Python package dependencies
- **Key Dependencies**:
  - `torch` (PyTorch) - Deep learning framework
  - `transformers` - HuggingFace transformers library
  - `datasets` - Dataset loading and processing
  - `librosa` - Audio processing
  - `gradio` - Web interface
  - `jiwer` - WER metric calculation
  - `whisper` - OpenAI Whisper model

### `CLAUDE.md`
- **Purpose**: Development guidelines and project metadata
- **Contains**: Technologies used, project structure, commands, code style

### `CPU_OPTIMIZATION_SUMMARY.md`
- **Purpose**: Summary of CPU optimization strategies for training
- **Contains**: Batch size reduction, memory optimization tips

---

## 🔧 `src/` - Main Source Code

### `src/__init__.py`
- Empty init file to make src a Python package

### **`src/cli/` - Command-Line Interface**

#### `src/cli/commands.py`
- **Purpose**: CLI commands for the system
- **Main Commands**:
  - `transcribe`: Transcribe audio to text
  - `demo`: Launch Gradio web interface
  - `evaluate`: Evaluate model performance
  - `batch`: Process multiple audio files
- **Usage**:
  ```bash
  python -m src.cli.commands transcribe audio.wav
  python -m src.cli.commands demo
  ```

#### `src/cli/__init__.py`
- CLI package initialization

---

### **`src/models/` - Data & ML Models**

#### `src/models/whisper_base.py` ⭐
- **Purpose**: OpenAI Whisper baseline model implementation
- **Key Features**:
  - Loads pre-trained Whisper models (tiny, base, small, medium, large)
  - Handles audio feature extraction
  - Performs transcription
  - Supports multiple languages (Arabic: 'ar')
- **Main Class**: `WhisperBaseModel`

#### `src/models/whisper_finetuner.py` ⭐
- **Purpose**: Fine-tuning Whisper on custom Arabic datasets
- **Key Features**:
  - Trains Whisper on MGB-2 or Common Voice datasets
  - Supports gradient accumulation for memory efficiency
  - Implements WER evaluation
  - Saves best checkpoints
- **Main Class**: `WhisperFinetuner`

#### `src/models/transcription_result.py`
- **Purpose**: Data model for transcription results
- **Contains**: TranscriptionResult class with:
  - Text output
  - Segments with timestamps
  - Confidence scores
  - Metadata

#### `src/models/segment.py`
- **Purpose**: Data model for audio segments
- **Contains**: Segment class with:
  - Start and end times
  - Segment text
  - Confidence score

#### `src/models/audio_file.py`
- **Purpose**: Audio file handling
- **Features**:
  - Load audio in multiple formats (WAV, MP3, FLAC, OGG, M4A)
  - Audio preprocessing
  - Resampling to target frequency

#### `src/models/base_model.py`
- **Purpose**: Abstract base class for all ASR models
- **Defines**: Contract that all models must implement

#### `src/models/serialization.py`
- **Purpose**: Serialization/deserialization of models and results
- **Features**:
  - Save models to disk
  - Load models from disk
  - Export results to JSON/CSV

---

### **`src/services/` - Core Business Logic**

#### `src/services/transcription_service.py`
- **Purpose**: Abstract base class for transcription services
- **Defines**: Interface all transcription services must implement

#### `src/services/transcription_pipeline.py`
- **Purpose**: Main transcription pipeline orchestrator
- **Features**:
  - Chains audio loading → feature extraction → model inference → result formatting
  - Handles error management
  - Coordinates between different services

#### `src/services/audio_processor.py`
- **Purpose**: Audio signal processing
- **Features**:
  - Load audio files
  - Apply spectral features
  - Handle different sample rates
  - Audio augmentation

#### `src/services/batch_processor.py`
- **Purpose**: Batch processing of multiple audio files
- **Features**:
  - Parallel processing support
  - Progress tracking
  - Error handling per file
  - Batch export

#### `src/services/evaluation_service.py`
- **Purpose**: Model evaluation and metrics
- **Metrics**:
  - WER (Word Error Rate)
  - CER (Character Error Rate)
  - Confidence scores

#### `src/services/demo_service.py`
- **Purpose**: Service for Gradio web demo
- **Features**:
  - Handles file uploads
  - Real-time transcription
  - Result formatting for web display

#### `src/services/storage_service.py`
- **Purpose**: File I/O and storage management
- **Features**:
  - Save transcription results
  - Load models from storage
  - Checkpoints and model versioning

#### `src/services/data_loader.py`
- **Purpose**: Dataset loading and preprocessing
- **Supports**:
  - HuggingFace Datasets
  - Common Voice, MGB-2 Arabic
  - Custom datasets

#### `src/services/common_voice_dataset.py`
- **Purpose**: Specific handler for Mozilla Common Voice dataset
- **Features**:
  - Download and preprocess Common Voice
  - Language-specific handling

---

### **`src/utils/` - Utility Functions**

#### `src/utils/audio.py`
- **Purpose**: Audio utility functions
- **Features**:
  - Load/save audio
  - Convert formats
  - Extract features (MFCC, spectrograms)

#### `src/utils/gpu_config.py`
- **Purpose**: GPU configuration and detection
- **Features**:
  - Detect available GPU
  - Fallback to CPU
  - Memory management

#### `src/utils/logging.py`
- **Purpose**: Logging configuration
- **Features**:
  - Structured logging
  - Log levels (DEBUG, INFO, WARNING, ERROR)
  - File and console output

#### `src/utils/config.py`
- **Purpose**: Configuration management
- **Features**:
  - Load YAML configs
  - Model parameters
  - Training hyperparameters

#### `src/utils/exceptions.py`
- **Purpose**: Custom exception classes
- **Examples**:
  - `AudioLoadError`
  - `ModelLoadError`
  - `TranscriptionError`

---

## 📚 `tests/` - Unit Tests

#### `tests/conftest.py`
- **Purpose**: Pytest configuration and fixtures
- **Provides**: Common test fixtures and setup

#### `tests/unit/test_whisper.py`
- **Purpose**: Unit tests for Whisper model
- **Tests**:
  - Model loading
  - Audio transcription
  - Output format validation

#### `tests/fixtures/test_audio.wav`
- **Purpose**: Sample audio file for testing

---

## 📓 `notebooks/` - Jupyter Notebooks

### ⭐ `notebooks/train_colab.ipynb`
- **Purpose**: Fine-tune Whisper on Google Colab (Free CPU)
- **Stages**:
  1. Install dependencies
  2. Mount Google Drive
  3. Load dataset (MGB-2 Arabic)
  4. Load and configure Whisper-tiny
  5. Preprocess audio and text
  6. Train with batch-wise processing
  7. Evaluate WER
  8. Save model to Drive
- **Key Features**:
  - Device-aware (CPU/GPU detection)
  - Memory-optimized (batch_size=1)
  - 18 epochs for better accuracy
  - 600 train / 100 validation samples
  - Uses HuggingFace token for faster downloads
  - Batch-wise training for stability

---

## 🎨 `demo/` - Web Interface

### `demo/app_gradio.py`
- **Purpose**: Gradio web interface for the system
- **Features**:
  - 4 tabs interface:
    1. **Transcription**: Single file upload and transcription
    2. **Batch Processing**: Multiple file processing
    3. **Model Comparison**: Compare different models
    4. **About**: Project information
  - Real-time results
  - Download results as JSON or ZIP
  - File format support: WAV, MP3, FLAC, OGG, M4A

---

## 📖 `docs/` - Documentation

### `docs/DEMO_GUIDE.md`
- Complete guide for using the web demo interface
- Screenshots and examples

### `docs/US3_FINETUNING.md`
- Guide for fine-tuning Whisper on custom datasets
- Step-by-step instructions

---

## 🛠️ `scripts/` - Standalone Scripts

### `scripts/download_models.py`
- Download pre-trained Whisper models
- Cache management

### `scripts/prepare_dataset.py`
- Download and prepare datasets (Common Voice, MGB-2)
- Audio preprocessing
- Dataset splitting

### `scripts/train_whisper.py`
- Full training script
- Hyperparameter tuning
- Model evaluation

---

## ⚙️ `config/` - Configuration Files

### `config/training_config.yaml`
- Training hyperparameters
- Model configuration
- Dataset paths

### `config/logging_config.yaml`
- Logging level settings
- Output formats

---

## 📋 `specs/` - Project Specifications

### `specs/main/`
- **plan.md**: Project plan and timeline
- **spec.md**: Technical specifications
- **tasks.md**: Task breakdown
- **research.md**: Research findings
- **quickstart.md**: Quick start guide

### `specs/main/contracts/`
- **CLI_CONTRACT.md**: CLI interface specification
- **DEMO_INTERFACE_CONTRACT.md**: Web demo interface contract

---

## 🎯 Key Python Files Explained

### Core Flow

1. **Entry Point**: `src/cli/commands.py`
   - CLI interface receives user input

2. **Pipeline**: `src/services/transcription_pipeline.py`
   - Orchestrates the transcription process

3. **Audio Loading**: `src/services/audio_processor.py` + `src/models/audio_file.py`
   - Loads and preprocesses audio

4. **Model Inference**: `src/models/whisper_base.py`
   - Runs Whisper model on audio

5. **Result Formatting**: `src/models/transcription_result.py`
   - Formats results with segments and timestamps

6. **Output**: `src/services/storage_service.py`
   - Saves results to disk (JSON, CSV, TXT)

### Training Flow

1. **Notebook**: `notebooks/train_colab.ipynb`
   - Interactive training on Colab

2. **Fine-Tuner**: `src/models/whisper_finetuner.py`
   - Training logic

3. **Data Loading**: `src/services/data_loader.py`
   - Load datasets

4. **Evaluation**: `src/services/evaluation_service.py`
   - Calculate WER and metrics

---

## 🚀 Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Test installation
python -c "import torch; print('GPU:', torch.cuda.is_available())"

# Transcribe single audio
python -m src.cli.commands transcribe audio.wav -o results.json

# Open web demo
python -m src.cli.commands demo

# Run tests
pytest tests/
```

---

## 📦 What's Included in ZIP

The `speech-to-text-project.zip` includes:
- ✅ All source code (`src/`)
- ✅ All configurations (`config/`)
- ✅ All documentation (`docs/`, `specs/`)
- ✅ Training notebooks (`notebooks/`)
- ✅ Demo interface (`demo/`)
- ✅ Scripts (`scripts/`)
- ✅ Tests (`tests/`)
- ✅ README and requirements
- ❌ Excluded: `__pycache__`, `.git`, large datasets, model checkpoints

---

## 📊 Project Statistics

- **Total Python Files**: ~15 core modules
- **Lines of Code**: ~3,000+ (main source)
- **Dependencies**: ~20+ packages
- **Supported Models**: Whisper (tiny, base, small, medium, large)
- **Languages**: Arabic (ar), extensible to other languages
- **Test Coverage**: Unit tests for core modules

---

## 🔑 Key Technologies

| Technology | Purpose |
|-----------|---------|
| **PyTorch** | Deep learning framework |
| **Transformers** | Pre-trained models (Whisper) |
| **Gradio** | Web interface |
| **Librosa** | Audio processing |
| **Datasets** | Dataset loading (HuggingFace) |
| **JIWER** | WER metric calculation |

---

## 📝 Usage Examples

### For GitHub README

Add this to your GitHub after uploaded:

```markdown
## Files Overview

- **src/**: Main codebase with CLI, models, services, and utilities
- **notebooks/train_colab.ipynb**: Fine-tune Whisper on Google Colab (Free CPU)
- **demo/app_gradio.py**: Web interface for transcription
- **scripts/**: Standalone training and data preparation scripts
- **tests/**: Unit tests
- **docs/**: Documentation and guides
- **configs/**: YAML configurations for training and logging

See PROJECT_STRUCTURE.md for detailed file-by-file explanation.
```

---

## 🎓 For Learning

If you're new to this project:
1. Start with `README.md`
2. Review `notebooks/train_colab.ipynb` to understand the flow
3. Check `src/models/whisper_base.py` to see model usage
4. Explore `src/services/transcription_pipeline.py` for orchestration
5. Review tests in `tests/` for usage examples

---

## 💡 Common Tasks

| Task | File to Edit |
|------|-------------|
| Change model size | `src/models/whisper_base.py` |
| Add new language | `src/utils/config.py` |
| Modify training params | `config/training_config.yaml` |
| Add new CLI command | `src/cli/commands.py` |
| Change demo interface | `demo/app_gradio.py` |
| Add evaluation metric | `src/services/evaluation_service.py` |

---

**Last Updated**: April 2026  
**Version**: 1.0  
**License**: MIT
