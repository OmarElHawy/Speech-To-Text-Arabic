# Quick Start Guide: Arabic Speech-to-Text System

**Version**: 1.0  
**Date**: 2026-04-05  
**Status**: Phase 1 Design Document

---

## Prerequisites

- **Python**: 3.9 or 3.10 (use Python 3.11 if available for better performance)
- **Git**: For version control
- **GPU** (optional but recommended): NVIDIA GPU with CUDA 11.8+ (e.g., RTX 3090, RTX 4090, A100)
  - **GPU Memory**: 6GB minimum (8GB+ recommended for fine-tuning)
  - **CPU Fallback**: Works without GPU but slower (10-30x slower inference)
- **RAM**: 16GB minimum system RAM
- **Disk Space**: 50GB (for models, dataset, and working files)

---

## Installation

### 1. Clone Repository & Navigate

```bash
cd "c:\it's me\E-JUST\NN\Speech to Text Project"
```

### 2. Create Python Virtual Environment

```bash
# Using venv (Python 3.9+)
python -m venv venv

# Activate
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# For notebook usage (optional)
pip install jupyter notebook ipykernel

# For demo interface
pip install gradio streamlit
```

### 4. Download Models & Dataset

```bash
# Download pretrained models (runs once, ~5GB)
python -m src.cli download-models

# Common Voice Arabic dataset should already exist in:
# cv-corpus-24.0-2025-12-05/ar/

# Verify dataset
python -m src.cli verify-dataset cv-corpus-24.0-2025-12-05/ar/
```

### 5. Verify Installation

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test basic transcription (short sample)
python -m src.cli transcribe --help
```

---

## Quick Test: Transcribe Your First Audio

### Option A: Using CLI

```bash
# Simple transcription (text output only)
python -m src.cli transcribe sample_audio.wav

# Full JSON output with all features
python -m src.cli transcribe sample_audio.wav \
  --json-output \
  --enable-speaker-id \
  --enable-emotion \
  --enable-keywords \
  --keywords "مرحبا, النظام, نص" \
  --output results.json

# Save output to file
python -m src.cli transcribe sample_audio.wav \
  --output transcript.txt
```

### Option B: Using Web Demo (Gradio)

```bash
# Start interactive demo
python -m src.cli demo

# Open browser to:
# http://localhost:7860
```

Then:
1. Upload an audio file (or record using microphone)
2. Select model (default: whisper-small-finetune)
3. Enable optional features (speaker ID, emotion, keywords)
4. Click "Transcribe" button
5. View results and download if needed

### Option C: Using Jupyter Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open notebook: notebooks/01_data_exploration.ipynb
# Run cells sequentially to:
# - Load and inspect audio samples
# - Visualize spectrograms
# - Test transcription pipeline
```

---

## Model Selection Guide

### Whisper-Small (Recommended)
- **Best for**: Accuracy + Speed balance
- **Inference time**: ~2 seconds per 10-second audio (GPU)
- **Accuracy**: WER ~18-20% after fine-tuning
- **Memory**: 4GB GPU minimum
- **Use**: Production, demos, batch processing

```bash
python -m src.cli transcribe audio.wav \
  --model whisper-small-finetune
```

### Whisper-Base
- **Best for**: Maximum accuracy
- **Inference time**: ~3-4 seconds per 10-second audio
- **Accuracy**: WER ~15-17% after fine-tuning
- **Memory**: 6GB GPU minimum
- **Use**: When accuracy is critical, offline demo

```bash
python -m src.cli transcribe audio.wav \
  --model whisper-base
```

### Wav2Vec 2.0 (XLSR)
- **Best for**: Multilingual evaluation
- **Inference time**: ~2-3 seconds per 10-second audio
- **Accuracy**: WER ~22-25% (varies by checkpoint)
- **Memory**: 5GB GPU minimum
- **Use**: Model comparison, research

```bash
python -m src.cli transcribe audio.wav \
  --model wav2vec2-xlsr
```

### DeepSpeech
- **Best for**: Lightweight, fast inference
- **Inference time**: ~1-2 seconds per 10-second audio (CPU)
- **Accuracy**: WER ~35-40% (lower accuracy)
- **Memory**: 2GB minimum (CPU friendly)
- **Use**: Baseline comparison, resource-constrained environments

```bash
python -m src.cli transcribe audio.wav \
  --model deepspeech
```

---

## Common Tasks

### Evaluate Model on Test Dataset

```bash
# Compute WER for a batch of files
python -m src.cli evaluate \
  --predictions predictions.json \
  --references references.json \
  --output results/wer_scores.json \
  --metrics wer,cer
```

**Input format** (predictions.json):
```json
[
  {"id": "sample_1", "text": "مرحبا بك في النظام"},
  {"id": "sample_2", "text": "كيف حالك اليوم"}
]
```

**Output format** (results/wer_scores.json):
```json
{
  "summary": {
    "wer": 0.18,
    "cer": 0.12,
    "total_samples": 2
  },
  "per_sample": [...]
}
```

### Benchmark All Models

```bash
# Run comprehensive comparison
python -m src.cli benchmark \
  --test-dataset cv-corpus-24.0-2025-12-05/ar \
  --models "whisper-small-finetune,whisper-base,wav2vec2-xlsr,deepspeech" \
  --output results/ \
  --num-samples 100
```

Creates: `results/benchmark_YYYY-MM-DD.json` with detailed metrics for each model.

### Process Audio with All Optional Features

```bash
python -m src.cli transcribe audio.wav \
  --enable-speaker-id \
  --enable-emotion \
  --enable-keywords \
  --keywords "الطوارئ,الموعد النهائي,الامتحان" \
  --json-output \
  --output full_results.json
```

**Full output includes**:
- Segments with timestamps
- Speaker identification (speaker 0, speaker 1, etc.)
- Emotion classification (happy, angry, neutral, sad)
- Detected keywords with confidence scores and timestamps

### Fine-tune Whisper on Arabic Data (Advanced)

```bash
# Prepare dataset
python scripts/prepare_dataset.py \
  --dataset-path cv-corpus-24.0-2025-12-05/ar \
  --output-dir data/processed

# Start fine-tuning
python scripts/train_whisper.py \
  --config config/training_config.yaml \
  --checkpoint whisper-small \
  --output-dir models/whisper-small-finetune \
  --num-epochs 5 \
  --batch-size 16
```

Expected training time: 10-20 hours on NVIDIA A100 (or 30-50 hours on GTX 3090)

---

## Configuration

### Using Config File

Create `~/.arabic_asr/config.yaml`:

```yaml
defaults:
  model: "whisper-small-finetune"
  device: "cuda"              # or cpu
  language: "ar"
  confidence_threshold: 0.0
  enable_speaker_id: false
  enable_emotion: false
  enable_keywords: false

inference:
  batch_size: 1
  timeout_seconds: 300
  gpu_memory_limit_mb: 4096

models:
  whisper-small-finetune:
    version: "v1.2.3"
```

Then use:
```bash
python -m src.cli transcribe audio.wav --config ~/.arabic_asr/config.yaml
```

### Environment Variables

```bash
# Override device
export DEVICE=cuda

# Override model
export MODEL=whisper-small-finetune

# Override language
export LANGUAGE=ar
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solutions**:
1. Reduce batch size: `--batch-size 1`
2. Use CPU: `--device cpu`
3. Use smaller model: `--model whisper-small` (instead of whisper-base)
4. Close other GPU applications (games, other ML tasks)

### Issue: "Model not found / Download failed"

**Solution**:
```bash
# Re-download models
python -m src.cli download-models --force

# Or download specific model
huggingface-cli download openai/whisper-small-en --cache-dir ./models
```

### Issue: "Audio file format not supported"

**Supported formats**: wav, mp3, flac, ogg, m4a

**Solution**: Convert your audio first
```bash
# Using ffmpeg
ffmpeg -i input.xyz -acodec libmp3lame -ab 192k output.mp3
```

### Issue: "Slow inference on CPU"

**Expected performance**:
- GPU (RTX 3090): 2-5 seconds per 30-second audio
- CPU (i7-12700K): 30-60 seconds per 30-second audio

**Optimization**:
1. Upgrade to GPU if possible
2. Use faster model: `--model whisper-small` or `deepspeech`
3. Batch process multiple files to amortize overhead

### Issue: "Demo interface not loading"

**Solution**:
```bash
# Check if port 7860 is available
netstat -an | grep 7860

# Use different port
python -m src.cli demo --port 7861
```

---

## Project Structure

```
.
├── src/                    # Main source code
│   ├── models/            # Model implementations
│   ├── services/          # Business logic
│   ├── cli/               # Command-line interface
│   └── utils/             # Utility functions
│
├── tests/                 # Unit and integration tests
│   ├── unit/
│   ├── integration/
│   └── fixtures/          # Test data
│
├── notebooks/             # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_comparison.ipynb
│   ├── 03_whisper_finetuning.ipynb
│   └── ...
│
├── demo/                  # Demo interface code
│   ├── app_gradio.py
│   └── app_streamlit.py
│
├── scripts/               # Utility scripts
│   ├── prepare_dataset.py
│   ├── train_whisper.py
│   └── evaluate_models.py
│
├── results/               # Generated results and benchmarks
├── docs/                  # Documentation
├── requirements.txt       # Python dependencies
└── README.md             # Project overview
```

---

## Next Steps

### For Users
1. [✓] Install dependencies
2. [✓] Test with Quick Test section above
3. [✓] Explore models with Model Selection Guide
4. → Read [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for advanced usage

### For Developers
1. [✓] Setup development environment
2. [ ] Read [ARCHITECTURE.md](ARCHITECTURE.md) for system design
3. [ ] Run integration tests: `pytest tests/integration/`
4. [ ] Explore fine-tuning notebooks for model training
5. → Read [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for contributing

### For Researchers
1. [✓] Review [MODEL_SELECTION.md](specs/main/research.md) for model rationale
2. [ ] Explore `notebooks/02_model_comparison.ipynb` for benchmarking
3. [ ] Check `notebooks/03_whisper_finetuning.ipynb` for training details
4. [ ] Review `results/benchmark_*.json` for SOTA comparisons

---

## Support & Feedback

- **Documentation**: See [docs/](docs/) directory
- **Issues/Bugs**: GitHub Issues (if applicable)
- **Questions**: Check [FAQ.md](docs/FAQ.md)

---

## Citation

If you use this system in research, please cite:

```bibtex
@misc{arabic_asr_2026,
  title={Deep Learning Based Arabic Audio Understanding and Retrieval System},
  author={Your Name},
  year={2026},
  publisher={GitHub},
  howpublished={\url{https://github.com/[...]/}}
}
```

Also cite the datasets/models used:
- OpenAI Whisper: [Radford et al., 2023](https://arxiv.org/abs/2212.04356)
- Mozilla Common Voice: [Ardila et al., 2020](https://arxiv.org/abs/1912.06032)
- Wav2Vec 2.0: [Baevski et al., 2020](https://arxiv.org/abs/2006.11477)

