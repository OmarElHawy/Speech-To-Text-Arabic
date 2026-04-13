# Data Model & Domain Entities

**Phase**: 1 (Design)  
**Generated**: 2026-04-05  
**Purpose**: Define data structures, relationships, validation rules, and state transitions for the Arabic speech-to-text system

## Core Entities

### AudioFile

**Purpose**: Represents an input audio file awaiting transcription.

**Attributes**:
- `id` (str, UUID): Unique identifier for the audio file
- `filename` (str): Original filename  
- `filepath` (str): Path to stored audio file
- `format` (str, enum): Audio format (wav, mp3, flac, ogg, m4a)
- `duration_seconds` (float): Length of audio in seconds
- `sample_rate` (int): Audio sample rate in Hz (e.g., 16000, 44100)
- `channels` (int): Number of audio channels (1=mono, 2=stereo)
- `file_size_bytes` (int): Size in bytes
- `uploaded_at` (datetime): Timestamp when file was uploaded/created
- `metadata` (dict, optional): Custom metadata (speaker name, source, etc.)

**Validation Rules**:
- `duration_seconds` must be > 0 and ≤ 3600 (max 1 hour for v1)
- `sample_rate` must be 8000, 16000, 44100, or 48000 Hz
- `channels` must be 1 or 2
- `format` must be in supported list: [wav, mp3, flac, ogg, m4a]
- `file_size_bytes` must be ≤ 500MB

**Relationships**:
- **has_many**: TranscriptionResult (1 file → multiple attempts/versions)
- **has_many**: AudioSegment (via TranscriptionResult)

**State Transitions**:
```
UPLOADED → PROCESSING → COMPLETED / FAILED
```

---

### TranscriptionResult

**Purpose**: Output of speech-to-text processing for a single audio file.

**Attributes**:
- `id` (str, UUID): Unique identifier
- `audio_file_id` (str, FK): Reference to source AudioFile
- `model_name` (str): Which model generated this (e.g., "whisper-small-finetune", "wav2vec2-xlsr", "deepspeech")
- `model_version` (str): Model version/checkpoint identifier
- `text` (str): Full transcribed text in Arabic
- `confidence_score` (float, 0-1): Overall confidence in transcription
- `word_error_rate` (float, optional): WER if reference transcript available
- `processing_time_ms` (int): Inference duration in milliseconds
- `gpu_used` (bool): Whether GPU was used for inference
- `created_at` (datetime): When transcription was generated
- `is_manual_override` (bool): Whether human corrected this result

**Validation Rules**:
- `confidence_score` must be between 0.0 and 1.0
- `word_error_rate` must be between 0.0 and 1.0 if provided
- `processing_time_ms` must be > 0
- `text` must not be empty unless status is FAILED
- `model_name` must be in approved list (enforces model versioning)

**Relationships**:
- **belongs_to**: AudioFile
- **has_many**: Segment (one transcription has multiple segments)
- **has_one**: EvaluationMetrics (optional)

**State Transitions**:
```
QUEUED → PROCESSING → COMPLETED / FAILED
         ↑
         └─ (retry on FAILED)
```

---

### Segment

**Purpose**: Temporal division of audio with fine-grained transcription and metadata.

**Attributes**:
- `id` (str, UUID): Unique identifier
- `transcription_result_id` (str, FK): Reference to parent TranscriptionResult
- `segment_index` (int): Order in the transcription (0, 1, 2, ...)
- `start_time_seconds` (float): Segment start time (0.0 for beginning)
- `end_time_seconds` (float): Segment end time
- `text` (str): Transcribed text for this segment
- `confidence_score` (float, 0-1): Confidence for this specific segment
- `speaker_id` (int, optional): Speaker label (0, 1, 2, ... for different speakers)
- `emotion_label` (str, optional): Emotion classification (happy, angry, neutral, sad)
- `emotion_confidence` (float, optional): Confidence of emotion classification
- `detected_keywords` (list[Keyword], optional): Keywords found in this segment
- `created_at` (datetime): When segment was created

**Validation Rules**:
- `start_time_seconds` < `end_time_seconds`
- `end_time_seconds` - `start_time_seconds` ≥ 0.1 (minimum 100ms segments)
- `confidence_score` must be 0.0-1.0
- `emotion_label` must be in [happy, angry, neutral, sad] if provided
- `speaker_id` must be ≥ 0
- `text` must not be empty

**Relationships**:
- **belongs_to**: TranscriptionResult
- **has_one**: SpeakerLabel (optional, implicit in speaker_id)
- **has_one**: EmotionLabel (optional, implicit in emotion fields)
- **has_many**: KeywordDetection

**Nested Objects**:
```json
{
  "detected_keywords": [
    {"keyword": "emergency", "confidence": 0.98, "position": 5},
    {"keyword": "deadline", "confidence": 0.92, "position": 12}
  ]
}
```

---

### SpeakerLabel

**Purpose**: Metadata about speaker identity in a segment.

**Attributes**:
- `id` (str, UUID): Unique identifier
- `segment_id` (str, FK): Reference to Segment
- `speaker_id` (int): Numeric speaker identifier (0, 1, 2, ...)
- `speaker_name` (str, optional): Human-readable speaker name if known
- `confidence_score` (float, 0-1): Confidence in speaker identification
- `speaker_embedding` (list[float], optional): Vector representation of speaker voice

**Validation Rules**:
- `speaker_id` must be ≥ 0
- `confidence_score` must be 0.0-1.0
- `speaker_embedding` if provided must have consistent dimension (e.g., 256 floats)

**Relationships**:
- **belongs_to**: Segment

---

### EmotionLabel

**Purpose**: Emotion classification for a segment.

**Attributes**:
- `id` (str, UUID): Unique identifier
- `segment_id` (str, FK): Reference to Segment  
- `emotion_class` (str, enum): Classification [happy, angry, neutral, sad]
- `confidence_score` (float, 0-1): Confidence in emotion classification
- `model_used` (str): Which emotion model was used

**Validation Rules**:
- `emotion_class` must be exactly one of: [happy, angry, neutral, sad]
- `confidence_score` must be 0.0-1.0
- Each Segment has at most 1 EmotionLabel

**Relationships**:
- **belongs_to**: Segment

---

### KeywordDetection

**Purpose**: Records when a configured keyword is detected in a segment or full transcription.

**Attributes**:
- `id` (str, UUID): Unique identifier
- `segment_id` (str, FK): Reference to Segment (if segment-level detection)
- `transcription_result_id` (str, optional, FK): Reference to full result (if result-level detection)
- `keyword` (str): The detected keyword (e.g., "emergency", "deadline")
- `position_in_text` (int): Character position in segment/transcription text
- `confidence_score` (float, 0-1): Confidence in keyword match
- `detection_method` (str): How detected (text_match, phonetic, fuzzy_match, etc.)

**Validation Rules**:
- `keyword` must not be empty
- `confidence_score` must be 0.0-1.0
- `position_in_text` must be ≥ 0
- Either `segment_id` OR `transcription_result_id` must be set, not both

**Relationships**:
- **belongs_to**: Segment (optional)
- **belongs_to**: TranscriptionResult (optional)

---

### EvaluationMetrics

**Purpose**: Performance metrics for a TranscriptionResult when reference text is available.

**Attributes**:
- `id` (str, UUID): Unique identifier
- `transcription_result_id` (str, FK): Reference to TranscriptionResult
- `reference_text` (str): Ground-truth transcript (for comparison)
- `word_error_rate` (float, 0-1): WER metric
- `character_error_rate` (float, 0-1): CER metric  
- `match_error_rate` (float, 0-1): MER metric
- `word_information_preserved` (float, 0-1): WIP metric
- `num_insertions` (int): Insertion count in edit distance
- `num_deletions` (int): Deletion count in edit distance
- `num_substitutions` (int): Substitution count in edit distance
- `computed_at` (datetime): When metrics were calculated

**Validation Rules**:
- All rate metrics must be 0.0-1.0
- `num_*` counts must be ≥ 0
- Sum of insertions/deletions/substitutions should match computed edit distance

**Relationships**:
- **belongs_to**: TranscriptionResult

---

## Relationships Summary

```
AudioFile (1) ──→ (N) TranscriptionResult
                          │
                          ├──→ (N) Segment
                          │         ├──→ (1) SpeakerLabel [optional]
                          │         ├──→ (1) EmotionLabel [optional]
                          │         └──→ (N) KeywordDetection
                          │
                          └──→ (1) EvaluationMetrics [optional]
```

**Cardinality Notes**:
- 1 AudioFile can have multiple TranscriptionResults (different models, retries, versions)
- 1 TranscriptionResult has N Segments (one per time interval)
- 1 Segment can have at most 1 SpeakerLabel and 1 EmotionLabel
- 1 Segment can have N KeywordDetections (different keywords)
- 1 TranscriptionResult optionally has 1 EvaluationMetrics

---

## Data Persistence Strategy

### Primary Storage
- **Transcription metadata & results**: JSON files (for version control, demos) or SQLite (for production search)
- **Audio files**: Reference paths (don't duplicate)
- **Model artifacts**: Checkpoint directories for trained models

### Secondary Storage
- **Evaluation results**: CSV for analysis, JSON for archival
- **Speaker embeddings**: NumPy arrays (.npy files) for efficient storage
- **Cached spectrograms**: Optional temp files for training efficiency

### Serialization Format

**JSON Example** (TranscriptionResult + Segments):
```json
{
  "id": "tr-2026-0405-001",
  "audio_file_id": "af-2026-0405-001",
  "model_name": "whisper-small-finetune",
  "model_version": "v1.2.3",
  "text": "مرحبا بك في نظام تحويل الكلام إلى نص",
  "confidence_score": 0.92,
  "word_error_rate": 0.15,
  "processing_time_ms": 2450,
  "created_at": "2026-04-05T14:30:00Z",
  "segments": [
    {
      "id": "seg-001",
      "segment_index": 0,
      "start_time_seconds": 0.0,
      "end_time_seconds": 3.5,
      "text": "مرحبا بك",
      "confidence_score": 0.95,
      "speaker_id": 0,
      "emotion_label": "neutral",
      "emotion_confidence": 0.88,
      "detected_keywords": [
        {"keyword": "مرحبا", "confidence": 0.99, "position": 0}
      ]
    },
    {
      "id": "seg-002",
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
  ]
}
```

---

## State Management & Workflows

### Transcription Processing Workflow

```
1. AudioFile.UPLOADED
   ↓
2. TranscriptionResult.QUEUED
   ↓
3. Audio preprocessing (resample, normalize)
   ↓
4. Model inference (ASR pipeline)
   ↓
5. Segment generation (chunking by timestamps or pauses)
   ↓
6. Speaker diarization (if enabled)
   ↓
7. Emotion classification (if enabled)
   ↓
8. Keyword spotting (if enabled)
   ↓
9. TranscriptionResult.COMPLETED
   ↓
10. Result serialization (JSON/DB) & notification
```

### Error Handling

```
TranscriptionResult.PROCESSING
  ↓
[Error during inference]
  ↓
TranscriptionResult.FAILED
  ↓
[Log error, create alert]
  ↓
[Manual intervention or retry]
```

**Trackable error states**:
- `INVALID_FORMAT` - Unsupported audio format
- `CORRUPTED_FILE` - Audio file cannot be read
- `OUT_OF_MEMORY` - GPU/CPU memory exhausted
- `TIMEOUT` - Processing exceeded time limit
- `MODEL_ERROR` - Model loading or inference failed
- `UNKNOWN` - Unexpected error

---

## Validation & Constraints

### Cross-entity Constraints

1. **Temporal Consistency**: All Segment `start_time` and `end_time` must be within AudioFile `duration_seconds`
2. **Segment Ordering**: Segments must be ordered chronologically with no gaps > 2 seconds
3. **Speaker Continuity**: Speaker IDs must be assigned sequentially (0, 1, 2, ...) without gaps
4. **Metric Validity**: If WER is computed, reference text must be available and match language

### Business Rules

1. Only one "primary" TranscriptionResult per AudioFile per model  
2. Cannot delete AudioFile if it has uncompleted TranscriptionResults
3. Segments smaller than 100ms should be merged with adjacent segments
4. Empty detection lists are valid (audio with no detected keywords)

---

## Performance Considerations

- **Indexing**: `(audio_file_id, created_at)` on TranscriptionResult for efficient lookup
- **Denormalization**: Store `emotion_label` and `emotion_confidence` directly on Segment (avoid join for queries)
- **Pagination**: Use `segment_index` for efficient pagination of large results
- **Caching**: Cache model embeddings/checkpoints to avoid repeated downloads

