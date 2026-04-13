# Implementation Plan: Deep Learning Based Arabic Audio Understanding and Retrieval System

**Branch**: `main` | **Date**: 2026-04-05 | **Spec**: [specs/main/spec.md](specs/main/spec.md)
**Input**: Feature specification from `/specs/main/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Build a comprehensive Arabic speech-to-text system using deep learning (primarily OpenAI Whisper fine-tuned on Mozilla Common Voice Arabic data). The system combines Speech Recognition (ASR) with optional advanced capabilities: speaker identification, emotion detection, and keyword spotting. Deliverables include source code, architecture diagrams, experiments/results, evaluation metrics (WER), and a Gradio/Streamlit demo interface for interactive testing.

## Technical Context

**Language/Version**: Python 3.9+ (PyTorch, deep learning frameworks)  
**Primary Dependencies**: 
- OpenAI Whisper (primary ASR model)
- Wav2Vec 2.0 (model comparison)
- DeepSpeech (model comparison)
- PyTorch/TensorFlow for model training
- Librosa/torchaudio for audio processing
- Hugging Face Transformers library
- PyDiarize or Speechbrain for speaker diarization
- Emotion detection: HuBERT fine-tuned models or custom emotion classifier
- Keyword spotting: NEEDS CLARIFICATION (use CNN-based KWS or small model like MobileNet?)

**Storage**: 
- Mozilla Common Voice Arabic dataset (audio files + metadata TSV files)
- Training artifacts: model checkpoints, logs
- Results: JSON/CSV for WER metrics, confusion matrices
- Transcripts and metadata: searchable storage (JSON/SQLite for demo phase)

**Testing**: 
- Unit tests: model loading, inference on sample audio (pytest)
- Integration tests: full pipeline (audio → transcript → optional features)
- Evaluation: WER computation using jiwer library, benchmarking scripts

**Target Platform**: Linux/Windows servers, GPU-enabled (CUDA) or CPU fallback  
**Project Type**: Python library + CLI tool + Web demo (Gradio/Streamlit)  
**Performance Goals**: 
- ASR: Real-time or near-real-time (≥ 0.5x real-time speed on GPU)
- WER target: ≤ 20% on validation data
- Inference latency: ≤ 2 seconds for 10-second audio on GPU

**Constraints**: 
- Model size: Whisper-base/small preferred for faster inference
- Memory: < 16GB GPU VRAM for model + inference
- Training time: NEEDS CLARIFICATION (acceptable training duration?)
- Batch processing vs. streaming inference: NEEDS CLARIFICATION (preference?)

**Scale/Scope**: 
- Audio duration: 10 seconds to 1 hour per file
- Concurrent users: Single-user demo phase (scalability for future)
- Dataset size: ~30+ hours of Arabic speech in Common Voice collection

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

**Project Constitution Principles** (from `.specify/memory/constitution.md`):

1. **Research-First Design**: All technical choices (model selection, architecture, dependencies) must be justified with documented research or benchmarks. Avoid assumption-based decisions.
   - **Status (Phase 0)**: ✅ PASS - Spec requires model comparison
   - **Status (Phase 1)**: ✅ PASS - All technical choices documented in [research.md](research.md) with detailed rationale:
     * Whisper-small selected over alternatives with confidence assessment
     * HuBERT for emotion detection justified via transfer learning benefits
     * Text-based keyword spotting justified by >95% precision achievable
     * Pyannote.audio v3 for speaker diarization (SOTA, 85-90% accuracy)

2. **Test-Driven Development**: Implementation must be preceded by test specifications. All core functionality must have unit and integration tests.
   - **Status (Phase 0)**: ✅ PASS - Measurable success criteria defined
   - **Status (Phase 1)**: ✅ PASS - Project structure (tests/ directory) includes:
     * Unit tests for audio processing, model loading, transcription
     * Integration tests for full pipeline (audio → transcript → optional features)
     * Test fixtures with sample Arabic audio in tests/fixtures/
     * Evaluation specifications in [contracts/CLI_CONTRACT.md](contracts/CLI_CONTRACT.md#command-evaluate)

3. **Reproducibility & Documentation**: Models, datasets, training procedures, and evaluation scripts must be fully documented and reproducible.
   - **Status (Phase 0)**: ✅ PASS - Public dataset (Mozilla Common Voice Arabic)
   - **Status (Phase 1)**: ✅ PASS - Full documentation structure created:
     * [data-model.md](data-model.md) - Data persistence, serialization format, schema
     * [quickstart.md](quickstart.md) - Setup, configuration, installation instructions
     * scripts/ directory includes `prepare_dataset.py`, `train_whisper.py`, `evaluate_models.py`
     * notebooks/ provide reproducible fine-tuning and research workflows
     * config/ directory for hyperparameters and model configurations

4. **Independent Deliverables**: P1 features must be independently testable; optional features (P2) must not block core functionality.
   - **Status (Phase 0)**: ✅ PASS - P1/P2 clearly separated in spec
   - **Status (Phase 1)**: ✅ PASS - Project structure enforces separation:
     * **P1 (Must-Have)**: ASR (spec.md P1), model comparison (P1), Whisper fine-tuning (P1), demo interface (P1) ✓
     * **P2 (Optional)**: Speaker ID (P2), emotion detection (P2), keyword spotting (P2) ✓
     * Feature toggles in CLI (`--enable-speaker-id`, `--enable-emotion`, `--enable-keywords`)
     * Services architecture allows independent module loading/disabling

5. **Performance Accountability**: Performance requirements must be measurable and tracked. No vague success criteria.
   - **Status (Phase 0)**: ✅ PASS - Specific metrics defined
   - **Status (Phase 1)**: ✅ PASS - Comprehensive evaluation framework:
     * CLI contract specifies benchmark command with detailed output format
     * Data model includes EvaluationMetrics entity (WER, CER, MER, WIP)
     * Results structure specified in contracts for comparison charts
     * Quickstart includes evaluation instructions with expected outputs

**Re-checked Constitution Status**: ✅ **PASS** - All principles satisfied through Phase 1 design. Ready for Phase 2 implementation tasking.

## Project Structure

### Documentation (this feature)

```text
specs/main/
├── plan.md              # This file (created by /speckit.plan command)
├── spec.md              # Feature specification (created by /speckit.specify command)
├── research.md          # Phase 0 output: model research, benchmarking decisions
├── data-model.md        # Phase 1 output: data entities and relationships
├── quickstart.md        # Phase 1 output: setup and usage guide
└── contracts/           # Phase 1 output: API/CLI contracts
```

### Source Code (repository root)

```text
src/
├── models/              # Model implementations and fine-tuning
│   ├── __init__.py
│   ├── whisper_asr.py        # Whisper-based ASR pipeline
│   ├── wav2vec_asr.py        # Wav2Vec 2.0 baseline (comparison)
│   ├── deepspeech_asr.py     # DeepSpeech baseline (comparison)
│   ├── speaker_diarization.py # Speaker identification module
│   ├── emotion_classifier.py  # Emotion detection module
│   └── keyword_spotting.py    # Keyword detection module
│
├── services/            # Processing services
│   ├── __init__.py
│   ├── audio_processor.py     # Audio loading, preprocessing, resampling
│   ├── transcription_service.py # Orchestrates ASR pipeline
│   ├── evaluation_service.py  # WER computation, metric calculation
│   └── storage_service.py     # Transcript and metadata persistence
│
├── cli/                 # Command-line interface
│   ├── __init__.py
│   ├── main.py               # CLI entry point
│   └── commands.py           # CLI commands (transcribe, evaluate, benchmark)
│
└── utils/              # Utilities
    ├── __init__.py
    ├── config.py            # Configuration management
    └── logging.py           # Logging setup

tests/
├── unit/               # Unit tests for individual components
│   ├── test_audio_processor.py
│   ├── test_whisper_asr.py
│   ├── test_emotion_classifier.py
│   └── test_keyword_spotting.py
│
├── integration/        # Integration tests for full pipelines
│   ├── test_transcription_pipeline.py
│   └── test_evaluation.py
│
├── fixtures/          # Test data and mock files
│   ├── sample_arabic.wav
│   └── expected_transcripts.json
│
└── conftest.py       # Pytest configuration and fixtures

notebooks/
├── 01_data_exploration.ipynb      # Dataset analysis, audio samples
├── 02_model_comparison.ipynb       # Benchmark Whisper vs Wav2Vec vs DeepSpeech
├── 03_whisper_finetuning.ipynb     # Training fine-tuned Whisper model
├── 04_speaker_diarization.ipynb    # Speaker ID experiments
├── 05_emotion_detection.ipynb      # Emotion classification experiments
└── 06_keyword_spotting.ipynb       # Keyword spotting training and evaluation

demo/
├── app_gradio.py               # Gradio-based demo interface
├── app_streamlit.py            # Streamlit-based demo interface  (optional)
└── requirements_demo.txt       # Demo-specific dependencies

results/                         # Experiment outputs and evaluation results
├── model_comparison.json        # Benchmark results (Whisper vs Wav2Vec vs DeepSpeech)
├── wer_scores.csv              # WER results on test set
├── confusion_matrices/          # Emotion/speaker classification matrices
└── architecture_diagram.png    # System architecture visualization

config/
├── training_config.yaml         # Hyperparameters for fine-tuning
├── model_config.yaml            # Model-specific configurations
└── data_config.yaml             # Dataset paths and splits

data/                           # Data directory (cv-corpus already present)
├── cv-corpus-24.0-2025-12-05/
│   └── ar/
│       ├── train.tsv
│       ├── dev.tsv
│       ├── test.tsv
│       └── clips/
└── processed/                  # Preprocessed data for training

scripts/
├── download_dataset.sh          # Dataset download/verification script
├── prepare_dataset.py           # Data preprocessing and splits
├── train_whisper.py             # Fine-tuning script
├── evaluate_models.py           # Comprehensive evaluation script
└── predict.py                   # Inference script for single audio files

docs/
├── ARCHITECTURE.md              # System architecture and design decisions
├── USER_GUIDE.md                # Usage instructions
├── MODEL_SELECTION.md           # Model comparison rationale
├── SETUP.md                     # Development environment setup
└── API.md                       # API/CLI documentation

.github/workflows/               # CI/CD pipeline (future)
├── tests.yml
└── evaluate.yml

requirements.txt                 # Python dependencies
setup.py                        # Package setup
README.md                       # Project overview
.gitignore                      # Git ignore rules
```

**Structure Decision**: This structure follows a modular library design with clear separation between model implementations, services, CLI, and tests. The multi-notebook approach supports iterative research and hyperparameter exploration. Results and docs directories provide transparency for experiments and reproducibility. Demo apps are standalone to avoid dependency bloat for production use.


## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Three model implementations (Whisper, Wav2Vec, DeepSpeech) | Spec explicitly requires model comparison to justify final model selection and ensure optimal performance | Single model assumption would risk poor performance; comparative data justifies engineering effort |
| Optional P2 features (speaker ID, emotion, keyword spotting) | Spec marks these as "if time allows"; P1 (ASR + demo) is complete deliverable without them | Removing these limits feature set and future value; P2 status ensures they don't block core delivery |

**No constitution violations identified.** All design complexity is justified by feature requirements and independent delivery of P1/P2 features.

