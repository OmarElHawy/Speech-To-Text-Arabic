# Demo Interface Contract: Gradio/Streamlit UI

**Version**: 1.0  
**Technology**: Gradio 4.x (primary), Streamlit 1.x (optional alternative)  
**Input Protocol**: Audio file upload, text input, web form controls  
**Output Protocol**: HTML rendered interface, JSON API responses

---

## Gradio Interface Specification

### Layout & Components

#### Primary Tab: "Transcription"

**Purpose**: Convert audio to text with optional features.

**Input Components**:
1. **Audio Upload Box** (required)
   - Type: `gr.Audio()`
   - Accepts: Uploaded file or recording
   - Formats: wav, mp3, flac, ogg, m4a
   - Max size: 500MB
   - Max duration: 1 hour
   - Constraints: Clear button, sample audio links

2. **Model Selection** (required)
   - Type: `gr.Dropdown()`
   - Options: [whisper-small-finetune, whisper-base, wav2vec2-xlsr, deepspeech]
   - Default: whisper-small-finetune
   - Tooltip: "Whisper-small recommended. Fine-tuned on Arabic Common Voice."

3. **Feature Toggles** (optional, if time allows)
   - Type: `gr.Checkbox()`
   - Options:
     - ☐ Speaker Identification (detect multiple speakers)
     - ☐ Emotion Detection (happy/angry/neutral/sad)
     - ☐ Keyword Spotting (detect predefined keywords)

4. **Keyword List** (conditional, visible if "Keyword Spotting" enabled)
   - Type: `gr.Textbox()`
   - Label: "Keywords to detect (comma-separated, Arabic)"
   - Placeholder: "الطوارئ, الموعد النهائي, الامتحان"
   - Default: "المساعدة, الطوارئ, الموعد النهائي"

5. **Advanced Options** (optional, collapsible)
   - **Confidence Threshold** (slider 0.0-1.0, default 0.0)
   - **Language** (dropdown: Arabic [default], English, etc.)
   - **Device** (dropdown: Auto, GPU, CPU - default: Auto)

**Output Components**:
1. **Full Transcript** (read-only, scrollable textbox)
   - Type: `gr.Textbox()`
   - Content: Complete Arabic text transcript
   - Lines: Auto-adjusted to content
   - Copy button

2. **Confidence Score** (metric display)
   - Type: `gr.Number()`
   - Label: "Overall Confidence"
   - Format: "95.2%"

3. **Processing Time** (metric display)
   - Type: `gr.Number()`
   - Label: "Processing Time"
   - Format: "2.45 seconds"

4. **Segments Table** (if features enabled)
   - Type: `gr.Dataframe()`
   - Columns: 
     - Start Time (hh:mm:ss)
     - End Time (hh:mm:ss)
     - Speaker (if enabled)
     - Emotion (if enabled)
     - Text
     - Confidence
   - Sortable: By start time, confidence
   - Downloadable: CSV export

5. **Keywords Detected** (if keyword spotting enabled)
   - Type: `gr.HighlightedText()` or HTML
   - Display: Transcript with keywords highlighted in different colors
   - Show: Keyword, timestamp, confidence

**Buttons**:
- **Transcribe** (primary action button) - green
- **Clear** (reset button) - secondary
- **Download Results** (CSV/JSON export button) - secondary

---

#### Secondary Tab: "Batch Processing"

**Purpose**: Upload multiple audio files and process in batch.

**Input Components**:
1. **File Uploader** (multiple files)
   - Type: `gr.File(file_count="multiple")`
   - Accepts: Multiple audio files
   - Max total size: 2GB

2. **Processing Options** (same as single transcription)
   - Model selection, feature toggles, keywords

**Output Components**:
1. **Progress Bar**
   - Type: `gr.Progress()`
   - Shows: Files processed / total files, current file being processed

2. **Results Table**
   - Type: `gr.Dataframe()`
   - Columns:
     - Filename
     - Status (✓ completed, ✗ failed, ⏳ processing)
     - WER (if reference available)
     - Processing Time
     - Actions (view, download)

3. **Batch Download**
   - Type: `gr.Button()` or `gr.File()`
   - Output: ZIP file with all results (JSON + transcripts)

---

#### Tertiary Tab: "Model Comparison"

**Purpose**: View benchmarking results and model comparison.

**Input Components**:
1. **Dataset Selection** (dropdown)
   - Options: [Common Voice Arabic, Custom, Test Set]
   - Trigger: Auto-loads benchmark results when changed

2. **Metrics Toggle** (checkboxes)
   - ☐ WER (Word Error Rate)
   - ☐ CER (Character Error Rate)
   - ☐ Inference Time
   - ☐ Memory Usage

**Output Components**:
1. **Benchmark Comparison Chart**
   - Type: `gr.BarChart()` or Plotly integration
   - X-axis: Model names
   - Y-axis: WER/CER percentage (lower is better)
   - Color-coded: Green (best), Yellow (medium), Red (worst)

2. **Inference Time Comparison**
   - Type: `gr.LineChart()` or table
   - Shows: Latency by model on different hardware (GPU, CPU)

3. **Full Benchmark Table**
   - Type: `gr.Dataframe()`
   - Columns:
     - Model Name
     - Version
     - WER
     - CER
     - Avg Inference Time
     - GPU Memory
     - CPU Memory
     - Best For (fastest, most accurate, most efficient)

4. **Raw Results Download**
   - Type: `gr.Button()`
   - Output: JSON benchmark results

---

#### Quaternary Tab: "About & Help"

**Purpose**: Documentation and system info.

**Content**:
1. **System Information**
   - Model: [whisper-small-finetune v1.2.3]
   - PyTorch Version: [2.x.x]
   - CUDA Available: [Yes/No]
   - GPU: [NVIDIA RTX 4090 / None]

2. **Supported Languages**
   - Primary: Arabic (Modern Standard Arabic)
   - Secondary: English (fallback)

3. **Citation / References**
   - Links to:
     - OpenAI Whisper paper
     - Mozilla Common Voice Arabic dataset
     - Relevant academic papers

4. **Contact & Feedback**
   - Email link, issue tracker link, survey link

---

## Gradio API Endpoints

All components communicate via Gradio's internal API. External API users can call:

### POST `/api/predict`

**Request**:
```json
{
  "data": [
    "[audio_file_path or base64]",
    "whisper-small-finetune",
    true,
    true,
    true,
    "الطوارئ, الموعد النهائي, الامتحان",
    0.0,
    "ar",
    "auto"
  ]
}
```

**Response (Success)**:
```json
{
  "data": [
    "مرحبا بك في نظام تحويل الكلام إلى نص",
    0.92,
    2.45,
    [
      ["0.0-3.5", "0.0-3.5", "speaker 0", "neutral", "مرحبا بك", 0.95],
      ["3.5-8.2", "3.5-8.2", "speaker 0", "neutral", "في نظام تحويل الكلام إلى نص", 0.90]
    ],
    "الطوارئ: [0.0-3.5] confidence 0.99\nالموعد النهائي: [5.2-6.5] confidence 0.92"
  ]
}
```

**Response (Error)**:
```json
{
  "error": "INVALID_FORMAT",
  "message": "Unsupported audio format: xyz"
}
```

---

## Visual Design

### Color Scheme (Arabic-appropriate, accessible)
- **Primary**: Deep blue (#1E3A8A)
- **Success**: Green (#10B981)
- **Warning**: Orange (#F59E0B)
- **Error**: Red (#EF4444)
- **Background**: Light gray (#F3F4F6)
- **Text**: Dark gray (#1F2937)

### Typography
- **Header**: Arabic font (Droid Sans Arabic, 24px, bold)
- **Body**: Arabic font (Droid Sans Arabic, 14px, regular)
- **Code/Results**: Monospace (16px, for readability)

### Accessibility
- High contrast ratios (WCAG AA)
- All buttons have clear labels
- Recording components include visual/audio feedback
- Error messages are clear and actionable
- RTL support for Arabic text (Streamlit/Gradio native)

---

## User Workflows

### Workflow A: Single Audio Transcription
1. Upload audio (via file browser or microphone recording)
2. Select model (default: whisper-small-finetune)
3. (Optional) Enable speaker ID, emotion, keywords
4. Click "Transcribe" button
5. View results in transcript box + segments table
6. Download results as JSON/CSV if needed

### Workflow B: Batch Processing
1. Upload multiple audio files (drag-drop or file picker)
2. Configure options (model, features)
3. Click "Process Batch" button
4. Monitor progress in real-time
5. Download ZIP of all results

### Workflow C: Model Comparison
1. Navigate to "Model Comparison" tab
2. Run benchmark (first time) or view cached results
3. Select metrics of interest (WER, inference time, memory)
4. View charts and raw benchmark table
5. Download detailed results

---

## Performance Requirements

- **Initial Load Time**: < 5 seconds
- **Audio Upload**: < 2 seconds for 100MB file
- **Transcription**: Streaming results (first segment within 1 second)
- **Batch Processing**: Real-time progress updates (stream processing)
- **Model Comparison**: < 500ms to load cached benchmark results

---

## Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+
- Mobile browsers (iOS Safari 14+, Chrome Android 90+)

---

## Offline Mode (Optional Enhancement)

If offline support needed:
- Bundle model locally (requires additional setup)
- Cache benchmark results in browser local storage
- Disable network-dependent features (auth, logging)

