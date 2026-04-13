# CLI Contract: Arabic Speech-to-Text System

**Version**: 1.0  
**Format**: Command-line application with subcommands  
**Input Protocol**: JSON arguments, files, and stdin  
**Output Protocol**: JSON stdout, structured logging stderr  

---

## Command: `transcribe`

**Purpose**: Convert audio file(s) to text using fine-tuned Whisper model.

### Usage

```bash
python -m src.cli transcribe <audio_file> [OPTIONS]
```

### Options

```
--model TEXT                Model to use [default: whisper-small-finetune]
                           Choices: whisper-small-finetune, whisper-base, 
                                   wav2vec2-xlsr, deepspeech

--output TEXT              Output file path (JSON) [default: stdout]

--language TEXT            Language code [default: ar]

--enable-speaker-id        Enable speaker diarization [default: false]

--enable-emotion           Enable emotion detection [default: false]

--enable-keywords          Enable keyword spotting [default: false]

--keywords TEXT            Comma-separated keywords to detect
                           (e.g., "emergency,deadline,exam")

--device TEXT              Device to use (cuda, cpu) [default: auto]

--batch-size INT           Batch size for processing [default: 1]

--confidence-threshold FLOAT
                           Confidence threshold (0.0-1.0) [default: 0.0]

--json-output              Output full JSON (vs. text only) [default: false]

--progress                 Show progress bar [default: true]
```

### Input

- **File**: Single audio file in supported format (wav, mp3, flac, ogg, m4a)
- **Constraints**:
  - File size: ≤ 500MB
  - Duration: ≤ 1 hour
  - Sample rate: auto-resampled to 16kHz

### Output (Success)

**Default (text only)**:
```
مرحبا بك في نظام تحويل الكلام إلى نص
```

**With `--json-output`**:
```json
{
  "status": "success",
  "transcription": {
    "id": "tr-2026-0405-001",
    "text": "مرحبا بك في نظام تحويل الكلام إلى نص",
    "confidence_score": 0.92,
    "processing_time_ms": 2450,
    "model_name": "whisper-small-finetune",
    "model_version": "v1.2.3"
  },
  "segments": [
    {
      "segment_index": 0,
      "start_time_seconds": 0.0,
      "end_time_seconds": 3.5,
      "text": "مرحبا بك",
      "confidence_score": 0.95,
      "speaker_id": 0,
      "emotion_label": "neutral",
      "emotion_confidence": 0.88,
      "detected_keywords": [
        {"keyword": "مرحبا", "confidence": 0.99}
      ]
    },
    {
      "segment_index": 1,
      "start_time_seconds": 3.5,
      "end_time_seconds": 8.2,
      "text": "في نظام تحويل الكلام إلى نص",
      "confidence_score": 0.90,
      "speaker_id": 0,
      "emotion_label": "neutral",
      "emotion_confidence": 0.85,
      "detected_keywords": []
    }
  ],
  "metadata": {
    "audio_duration_seconds": 8.2,
    "language": "ar",
    "processing_time_ms": 2450,
    "gpu_used": true
  }
}
```

### Output (Error)

```json
{
  "status": "error",
  "error_code": "INVALID_FORMAT",
  "message": "Unsupported audio format: xyz",
  "details": {
    "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"],
    "provided_file": "audio.xyz"
  }
}
```

### Error Codes

| Code | HTTP | Meaning |
|------|------|---------|
| `SUCCESS` | 200 | Transcription completed successfully |
| `INVALID_FORMAT` | 400 | Unsupported audio format |
| `CORRUPTED_FILE` | 400 | Audio file cannot be read/parsed |
| `FILE_TOO_LARGE` | 413 | File exceeds 500MB size limit |
| `DURATION_EXCEEDED` | 413 | Audio duration > 1 hour |
| `INVALID_LANGUAGE` | 400 | Unsupported language code |
| `OUT_OF_MEMORY` | 507 | Insufficient GPU/CPU memory |
| `MODEL_ERROR` | 500 | Model loading or inference failed |
| `TIMEOUT` | 504 | Processing exceeded time limit (default: 300s) |
| `NOT_FOUND` | 404 | Model or file not found |
| `UNKNOWN` | 500 | Unexpected error |

---

## Command: `evaluate`

**Purpose**: Evaluate transcription accuracy against reference transcripts.

### Usage

```bash
python -m src.cli evaluate <predictions_file> <references_file> [OPTIONS]
```

### Input Files

**predictions_file** (JSON): 
```json
[
  {"id": "1", "text": "مرحبا بك في النظام"},
  {"id": "2", "text": "كيف حالك اليوم"}
]
```

**references_file** (JSON):
```json
[
  {"id": "1", "text": "مرحبا بك في التطبيق"},
  {"id": "2", "text": "كيف حالك اليوم"}
]
```

### Options

```
--output TEXT              Output file path (CSV/JSON) [default: stdout]

--format TEXT              Output format (csv, json) [default: json]

--metrics TEXT             Metrics to compute [default: wer,cer]
                           Choices: wer, cer, mer, wip

--detailed                 Include segment-level metrics [default: false]
```

### Output

```json
{
  "status": "success",
  "summary": {
    "total_samples": 2,
    "wer": 0.10,
    "cer": 0.05,
    "mer": 0.10,
    "wip": 0.90
  },
  "per_sample": [
    {
      "id": "1",
      "reference": "مرحبا بك في التطبيق",
      "prediction": "مرحبا بك في النظام",
      "wer": 0.25,
      "cer": 0.105,
      "insertions": 0,
      "deletions": 0,
      "substitutions": 1
    },
    {
      "id": "2",
      "reference": "كيف حالك اليوم",
      "prediction": "كيف حالك اليوم",
      "wer": 0.0,
      "cer": 0.0,
      "insertions": 0,
      "deletions": 0,
      "substitutions": 0
    }
  ]
}
```

---

## Command: `benchmark`

**Purpose**: Compare multiple models on test dataset.

### Usage

```bash
python -m src.cli benchmark <test_dataset> [OPTIONS]
```

### Input

**test_dataset**: Path to directory with:
- `audio/` - subdirectory with audio files
- `references.json` - ground-truth transcripts

### Options

```
--models TEXT              Models to benchmark [default: all]
                           Comma-separated list of:
                           whisper-small-finetune,whisper-base,
                           wav2vec2-xlsr,deepspeech

--output TEXT              Output directory [default: ./results/]

--num-samples INT          Max samples to run [default: 100]

--device TEXT              Device to use (cuda, cpu) [default: auto]
```

### Output

`results/benchmark_2026-04-05.json`:
```json
{
  "timestamp": "2026-04-05T14:30:00Z",
  "models": [
    {
      "model_name": "whisper-small-finetune",
      "model_version": "v1.2.3",
      "wer": 0.18,
      "cer": 0.12,
      "avg_inference_time_ms": 2150,
      "min_inference_time_ms": 1800,
      "max_inference_time_ms": 3200,
      "total_samples": 100,
      "failed_samples": 2,
      "gpu_memory_peak_mb": 3200
    },
    {
      "model_name": "wav2vec2-xlsr",
      "model_version": "huggingface-v4",
      "wer": 0.22,
      "cer": 0.15,
      "avg_inference_time_ms": 2400,
      "min_inference_time_ms": 2000,
      "max_inference_time_ms": 3400,
      "total_samples": 100,
      "failed_samples": 1,
      "gpu_memory_peak_mb": 4100
    },
    {
      "model_name": "deepspeech",
      "model_version": "mozilla-v1",
      "wer": 0.35,
      "cer": 0.28,
      "avg_inference_time_ms": 1800,
      "min_inference_time_ms": 1500,
      "max_inference_time_ms": 2200,
      "total_samples": 100,
      "failed_samples": 5,
      "gpu_memory_peak_mb": 2800
    }
  ],
  "winner": {
    "model_name": "whisper-small-finetune",
    "metric": "wer",
    "score": 0.18
  }
}
```

---

## Command: `demo`

**Purpose**: Start interactive Gradio web interface for transcription.

### Usage

```bash
python -m src.cli demo [OPTIONS]
```

### Options

```
--host TEXT                Server host [default: 0.0.0.0]

--port INT                 Server port [default: 7860]

--model TEXT               Model to use in demo [default: whisper-small-finetune]

--enable-all-features      Enable speaker ID, emotion, keywords [default: false]

--share                    Create public Gradio link [default: false]

--debug                    Run in debug mode [default: false]
```

### Output (stdout)

```
Gradio Interface loaded at:
  http://localhost:7860
  
Model: whisper-small-finetune (v1.2.3)
Device: cuda
Features enabled: speaker_id=false, emotion=false, keywords=false

Press Ctrl+C to stop
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error (check stderr) |
| 2 | Invalid arguments/config |
| 3 | File not found |
| 4 | Permission denied |
| 5 | Out of memory |
| 127 | Command not found |

---

## Error Message Format

All errors are output to stderr in JSON format:

```json
{
  "level": "error",
  "timestamp": "2026-04-05T14:30:00Z",
  "error_code": "MODEL_ERROR",
  "message": "Failed to load model checkpoint",
  "details": {
    "model": "whisper-small-finetune",
    "checkpoint": "/path/to/model.pt",
    "underlying_error": "CUDA out of memory"
  },
  "traceback": "[optional, only in debug mode]"
}
```

---

## Configuration File Support

Optional YAML config file at `~/.arabic_asr/config.yaml`:

```yaml
defaults:
  model: "whisper-small-finetune"
  device: "auto"
  language: "ar"
  confidence_threshold: 0.0
  enable_speaker_id: false
  enable_emotion: false
  enable_keywords: false

models:
  whisper-small-finetune:
    checkpoint: "/path/to/whisper-small-finetune.pt"
    version: "v1.2.3"
  
  wav2vec2-xlsr:
    huggingface_id: "facebook/wav2vec2-xlsr-53-arabic"

inference:
  batch_size: 1
  timeout_seconds: 300
  gpu_memory_limit_mb: 4096
```

Load with: `transcribe --config ~/.arabic_asr/config.yaml audio.wav`

