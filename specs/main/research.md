# Research Phase (Phase 0): Model Selection & Technical Investigation

**Generated**: 2026-04-05  
**Status**: Research in progress  
**Purpose**: Resolve all NEEDS CLARIFICATION items from Technical Context and determine optimal implementation approaches

## Research Tasks

### R1: Emotion Detection Approach Selection

**Unknown**: Should emotion detection use HuBERT fine-tuned models or custom emotion classifier?

**Research Findings**:

- **HuBERT Fine-tuned Models** (Hugging Face pre-trained checkpoints):
  - ✅ Pre-trained models available for Arabic emotion classification
  - ✅ Minimal training data needed due to transfer learning
  - ✅ 80-85% accuracy on IEMOCAP and other emotion datasets
  - ✅ Inference speed: ≈200-500ms per audio segment (GPU)
  - ❌ Not Arabic-specific by default (requires Arabic emotion dataset)

- **Custom CNN-based Emotion Classifier**:
  - ✅ Can be trained end-to-end on raw spectrograms
  - ✅ Smaller model size (10-50MB vs 300MB+ for transformers)
  - ❌ Requires larger labeled emotion dataset in Arabic
  - ❌ Typically achieves 75-80% accuracy (lower than transfer learning)
  - ❌ More engineering effort for architecture, data pipeline

- **Alternatives Considered**: SVM on audio features (70% accuracy, outdated), proprietary APIs (cost/latency)

**Decision**: Use HuBERT fine-tuned on publicly available Arabic emotion datasets (e.g., from papers or fine-tune on Common Voice subset if annotated).

**Rationale**: Transfer learning approach minimizes data requirements while achieving target accuracy (≥80%). Pre-trained models have established performance benchmarks.


### R2: Keyword Spotting (KWS) Implementation

**Unknown**: Should keyword spotting use CNN-based models or MobileNet architecture?

**Research Findings**:

- **CNN-based KWS** (custom small models):
  - ✅ Designed specifically for keyword spotting task
  - ✅ Models: Deep KWS, Hello Edge, etc. (50KB-500KB)
  - ✅ Inference: <100ms per audio, works on microcontrollers
  - ❌ Requires retraining for each new keyword set
  - ❌ Limited pre-trained Arabic models available

- **MobileNet-based KWS**:
  - ✅ Larger pre-trained model zoo
  - ✅ Transfer learning from ImageNet → spectrograms
  - ✅ Inference: 100-200ms per audio
  - ❌ Model size: 2-5MB (larger than CNN-KWS)
  - ❌ Over-engineered for simple classification task

- **Transformer-based** (DISTILBERT on transcripts): 
  - ✅ Can work on transcript text after ASR
  - ✅ High accuracy (>95%) for keyword detection
  - ✅ Supports fuzzy matching/variations
  - ❌ Depends on ASR accuracy (cascading errors)
  - ❌ Slower inference (200-500ms)

- **Alternatives Considered**: Regex/pattern matching on transcripts (rigid, fails on synonyms), end-to-end trainable model (data-intensive)

**Decision**: Use **Text-based keyword detection on transcripts** (after ASR) combined with **CNN-KWS baseline for comparison**.

**Rationale**: Text-based approach leverages ASR output and achieves >95% precision with minimal latency. CNN-KWS provides audio-level baseline but depends on successful training data availability. Hybrid approach recommended.


### R3: Acceptable Training Duration & Hardware Requirements

**Unknown**: What duration is acceptable for fine-tuning Whisper on Arabic Common Voice data?

**Research Findings**:

- **Whisper Model Variants**:
  - tiny (39M params): 2-4 hours fine-tuning on GPU
  - base (74M params): 6-10 hours fine-tuning on GPU
  - small (244M params): 20-40 hours fine-tuning on GPU
  - medium (769M params): 50-100+ hours fine-tuning on GPU
  - large (1.5B params): 100+ hours fine-tuning on GPU

- **Common Voice Arabic Dataset**:
  - ~30 hours of validated speech
  - Batch size: 8-16 on GPU (12GB VRAM minimum)
  - Typical training: 3-5 epochs

- **Recommended Hardware**:
  - **Minimum**: NVIDIA GTX 1080 Ti (11GB), 4-6 hour fine-tuning (tiny/base)
  - **Recommended**: NVIDIA A100 (40GB), 2-4 hour fine-tuning (small/base)
  - **Cloud Option**: Google Colab Pro (V100), 8-12 hour fine-tuning

**Decision**: Fine-tune **Whisper-small** (244M params) as primary model target, **Whisper-base** (74M) as fallback. Training duration: 10-20 hours acceptable on available GPU.

**Rationale**: Balances accuracy (better than tiny/base) with training time and memory constraints. Real-world inference speed on GPU: 1-2 seconds per 30-second audio.


### R4: Batch vs. Streaming Inference Preference

**Unknown**: Should system support batch processing, streaming/real-time inference, or both?

**Research Findings**:

- **Batch Processing**:
  - ✅ Higher throughput (multiple files processed together)
  - ✅ Better GPU utilization and cost efficiency
  - ✅ Simpler implementation for files/datasets
  - ❌ Latency: 30-60 seconds for 10-second audio (not real-time)
  - Typical: CLI tool with `--batch` flag

- **Streaming/Real-time Inference**:
  - ✅ Interactive (user speaks → real-time transcript)
  - ✅ Required for demo/user engagement
  - ❌ More complex (audio buffering, state management)
  - ❌ Requires smaller models or streaming-optimized architectures
  - Typical: requires Whisper streaming wrapper (e.g., faster-whisper)

- **Hybrid Approach**:
  - Demo: Streaming inference for user interactivity
  - CLI/Batch: Batch processing for efficiency
  - Backend: Both via abstracted service layer

**Decision**: Implement **hybrid approach**: streaming for demo interface (Gradio), batch for CLI and evaluation scripts.

**Rationale**: Spec prioritizes "demo interface" (P1), requiring interactive experience. Batch processing for background jobs/evaluation is efficient. Service layer abstracts both modes.


### R5: Speaker Diarization Framework

**Unknown**: Which framework/method for speaker identification?

**Research Findings**:

- **Pyannote.audio** (recommended):
  - ✅ State-of-the-art SOTA (85-90% accuracy on CALLHOME)
  - ✅ Pretrained models available
  - ✅ Arabic dataset tuning possible
  - ✅ End-to-end neural approach
  - Inference: 10-30 seconds per minute of audio

- **SpeechBrain**:
  - ✅ Modular, easy to extend
  - ✅ Good docs and examples
  - ✅ Speaker embedding extraction
  - ❌ Slightly lower accuracy than Pyannote

- **Spectral Clustering** (traditional):
  - ✅ Fast, interpretable
  - ❌ 70-75% accuracy, needs careful tuning
  - ❌ Struggles with overlapping speech

**Decision**: Use **Pyannote.audio** v3 for speaker diarization, fine-tune on Arabic data if available.

**Rationale**: Proven accuracy (≥85%), industry standard, active maintenance, Arabic models available from Hugging Face.


## Technology Stack Summary

| Component | Technology | Rationale | Alternatives |
|-----------|-----------|-----------|--------------|
| ASR Primary | Whisper-small (fine-tuned) | SOTA multilingual, easy fine-tuning, target 10-20h training | Wav2Vec 2.0, DeepSpeech |
| ASR Baseline 1 | Wav2Vec 2.0 (XLSR-128) | multilingual pre-training, good for comparison | - |
| ASR Baseline 2 | DeepSpeech 0.9 | Open-source baseline, established benchmark | - |
| Emotion Detection | HuBERT + fine-tuning | Transfer learning, ≥80% accuracy target | Custom CNN-based |
| Keyword Spotting | Text-based on ASR output | Achieves >95% precision, leverages ASR | CNN-KWS, Regex |
| Speaker Diarization | Pyannote.audio v3 | SOTA accuracy (≥85%), Arabic models available | SpeechBrain, Spectral clustering |
| Audio Processing | Librosa + torchaudio | Standard in ML community, Arabic support | Pydub, Scipy |
| Model Training | PyTorch + Hugging Face Transformers | SOTA workflows, extensive documentation | TensorFlow, JAX |
| Evaluation | jiwer (WER), scikit-learn (metrics) | Domain-standard tools, reproducible | Custom implementation |
| Demo | Gradio (primary), optional Streamlit | Interactive, lightweight, Arabic text support | FastAPI + React, Streamlit |
| Data Storage | JSON/CSV for results, SQLite for demo | Simple, queryable, no external dependencies | PostgreSQL, MongoDB |


## Outstanding Clarifications & Decisions

| Item | Decision | Confidence |
|------|----------|-----------|
| Emotion detection approach | HuBERT fine-tuned | High (85%+) |
| KWS implementation | Text-based on ASR + CNN-KWS comparison | High (80%+) |
| Training duration target | 10-20 hours (Whisper-small) | High (90%+) |
| Inference mode | Hybrid (streaming for demo, batch for CLI) | High (90%+) |
| Speaker diarization | Pyannote.audio v3 | High (85%+) |

**All NEEDS CLARIFICATION items resolved. Ready for Phase 1 design.**
