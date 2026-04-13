#!/usr/bin/env python3
"""
Gradio Web Interface for Arabic Speech-to-Text System

This module provides a web-based interface for the speech-to-text system with:
- Single file transcription
- Batch processing
- Model comparison
- About & Help

Author: Speech-to-Text Project Team
Version: 1.0.0
"""

import os
import sys
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.batch_processor import BatchProcessor
from services.demo_service import DemoService
from services.evaluation_service import EvaluationService
from models.transcription_result import TranscriptionResult
from utils.config import Config
from utils.logging import get_logger
from utils.exceptions import AudioProcessingError, ModelLoadError

# Initialize logger
logger = get_logger(__name__)

# Initialize services
config = Config()
demo_service = DemoService()
batch_processor = BatchProcessor()
evaluation_service = EvaluationService()

# Available models
AVAILABLE_MODELS = [
    "whisper-small",
    "whisper-base",
    "whisper-medium",
    "whisper-large-v2"
]

# Supported languages
SUPPORTED_LANGUAGES = {
    "ar": "العربية (Arabic)",
    "en": "English",
    "fr": "Français (French)",
    "es": "Español (Spanish)"
}


def create_transcription_tab() -> gr.Tab:
    """
    Create the main transcription tab for single audio file processing.

    Returns:
        gr.Tab: Configured transcription tab
    """
    with gr.TabItem("📝 Transcription", id="transcription") as tab:

        gr.Markdown("""
        # 🎤 Arabic Speech-to-Text Transcription
        Upload an audio file or record audio to convert speech to Arabic text using advanced AI models.
        """)

        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                audio_input = gr.Audio(
                    label="Audio File",
                    type="filepath"
                )

                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="whisper-small",
                    label="Model",
                    info="Select the AI model for transcription"
                )

                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="ar",
                    label="Language",
                    info="Primary language for transcription"
                )

                with gr.Accordion("Advanced Options", open=False):
                    confidence_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        label="Confidence Threshold",
                        info="Minimum confidence score for segments (0.0 = show all)"
                    )

                    device_dropdown = gr.Dropdown(
                        choices=["auto", "cpu", "cuda"],
                        value="auto",
                        label="Device",
                        info="Compute device for inference"
                    )

                # Action buttons
                with gr.Row():
                    transcribe_btn = gr.Button(
                        "🎯 Transcribe",
                        variant="primary",
                        size="lg"
                    )
                    clear_btn = gr.Button(
                        "🗑️ Clear",
                        variant="secondary"
                    )

            with gr.Column(scale=1):
                # Output section
                with gr.Group():
                    gr.Markdown("### 📄 Transcript")
                    transcript_output = gr.Textbox(
                        label="Full Transcript",
                        lines=8,
                        interactive=False
                    )

                with gr.Row():
                    confidence_display = gr.Number(
                        label="Confidence Score",
                        minimum=0.0,
                        maximum=1.0,
                        precision=3
                    )
                    time_display = gr.Number(
                        label="Processing Time (s)",
                        minimum=0.0,
                        precision=2
                    )

                # Segments table
                segments_table = gr.Dataframe(
                    headers=["Start", "End", "Text", "Confidence"],
                    label="Transcription Segments",
                    interactive=False
                )

                # Download button
                download_btn = gr.File(
                    label="Download Results",
                    type="filepath",
                    visible=False
                )

        # Event handlers
        def transcribe_audio(
            audio_file: str,
            model: str,
            language: str,
            confidence_threshold: float,
            device: str
        ) -> Tuple[str, float, float, List[List], str]:
            """Handle audio transcription."""
            if not audio_file:
                raise gr.Error("Please upload an audio file or record audio")

            try:
                logger.info(f"Starting transcription: audio_file={audio_file}, model={model}, language={language}")
                start_time = time.time()

                # Perform transcription
                result = demo_service.transcribe_audio(
                    audio_path=audio_file,
                    model_name=model,
                    language=language,
                    confidence_threshold=confidence_threshold,
                    device=device
                )

                processing_time = time.time() - start_time
                logger.info(f"Transcription completed in {processing_time:.2f}s")
                logger.debug(f"Result text: {result.text[:100] if result.text else 'None'}")
                logger.debug(f"Confidence: {result.confidence_score}, Segments: {len(result.segments or [])}")

                # Format segments for display
                segments_data = []
                if result.segments:
                    for segment in result.segments:
                        segments_data.append([
                            f"{segment.start_time:.2f}",
                            f"{segment.end_time:.2f}",
                            segment.text,
                            f"{segment.confidence_score:.3f}" if segment.confidence_score else "0.000"
                        ])

                # Save results to temporary file for download
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    delete=False,
                    encoding='utf-8'
                )
                json.dump(result.to_dict(), temp_file, indent=2, ensure_ascii=False)
                temp_file.close()

                logger.info(f"Transcription result saved to {temp_file.name}")

                return (
                    result.text or "",
                    result.confidence_score or 0.0,
                    processing_time,
                    segments_data,
                    temp_file.name
                )

            except Exception as e:
                error_msg = f"Transcription failed: {str(e)}"
                logger.error(error_msg, exc_info=True)
                raise gr.Error(error_msg)

        def clear_inputs():
            """Clear all inputs and outputs."""
            return (
                None,  # audio_input
                "whisper-small",  # model_dropdown
                "ar",  # language_dropdown
                0.0,  # confidence_slider
                "auto",  # device_dropdown
                "",  # transcript_output
                0.0,  # confidence_display
                0.0,  # time_display
                [],  # segments_table
                None  # download_btn
            )

        # Wire up event handlers
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[
                audio_input,
                model_dropdown,
                language_dropdown,
                confidence_slider,
                device_dropdown
            ],
            outputs=[
                transcript_output,
                confidence_display,
                time_display,
                segments_table,
                download_btn
            ]
        )

        clear_btn.click(
            fn=clear_inputs,
            outputs=[
                audio_input,
                model_dropdown,
                language_dropdown,
                confidence_slider,
                device_dropdown,
                transcript_output,
                confidence_display,
                time_display,
                segments_table,
                download_btn
            ]
        )

    return tab


def create_batch_processing_tab() -> gr.Tab:
    """
    Create the batch processing tab for multiple audio files.

    Returns:
        gr.Tab: Configured batch processing tab
    """
    with gr.TabItem("📦 Batch Processing", id="batch") as tab:

        gr.Markdown("""
        # 📦 Batch Audio Processing
        Upload multiple audio files for batch transcription processing.
        """)

        with gr.Row():
            with gr.Column():
                # Input section
                file_uploader = gr.File(
                    file_count="multiple",
                    type="filepath",
                    file_types=[".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"],
                    label="Audio Files",
                    height=200
                )

                model_dropdown = gr.Dropdown(
                    choices=AVAILABLE_MODELS,
                    value="whisper-small",
                    label="Model"
                )

                language_dropdown = gr.Dropdown(
                    choices=list(SUPPORTED_LANGUAGES.keys()),
                    value="ar",
                    label="Language"
                )

                # Process button
                process_btn = gr.Button(
                    "🚀 Process Batch",
                    variant="primary",
                    size="lg"
                )

            with gr.Column():
                # Progress section
                progress_bar = gr.Progress()
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=3
                )

                # Results table
                results_table = gr.Dataframe(
                    headers=["Filename", "Status", "Confidence", "Time (s)", "Actions"],
                    label="Processing Results",
                    interactive=False
                )

                # Batch download
                batch_download = gr.File(
                    label="Download All Results (ZIP)",
                    visible=False
                )

        # Event handlers
        def process_batch(
            files: List[str],
            model: str,
            language: str,
            progress: gr.Progress = gr.Progress()
        ) -> Tuple[str, List[List], str]:
            """Process multiple audio files in batch."""
            if not files:
                raise gr.Error("Please upload audio files")

            try:
                total_files = len(files)
                results = []

                for i, file_path in enumerate(files, 1):
                    progress(i / total_files, f"Processing {i}/{total_files}: {os.path.basename(file_path)}")

                    try:
                        # Process individual file
                        result = demo_service.transcribe_audio(
                            audio_path=file_path,
                            model_name=model,
                            language=language
                        )

                        results.append({
                            "filename": os.path.basename(file_path),
                            "status": "✓ Completed",
                            "confidence": ".3f",
                            "time": ".2f",
                            "result": result
                        })

                    except Exception as e:
                        results.append({
                            "filename": os.path.basename(file_path),
                            "status": f"✗ Failed: {str(e)}",
                            "confidence": 0.0,
                            "time": 0.0,
                            "result": None
                        })

                # Create ZIP file with all results
                import zipfile
                zip_path = tempfile.mktemp(suffix='.zip')
                with zipfile.ZipFile(zip_path, 'w') as zf:
                    for item in results:
                        if item["result"]:
                            # Add JSON result
                            json_filename = f"{Path(item['filename']).stem}_result.json"
                            zf.writestr(json_filename, json.dumps(
                                item["result"].to_dict(),
                                indent=2,
                                ensure_ascii=False
                            ))

                # Format table data
                table_data = [
                    [
                        item["filename"],
                        item["status"],
                        item["confidence"],
                        item["time"],
                        "Download" if item["result"] else ""
                    ]
                    for item in results
                ]

                status_msg = f"Processed {len(results)} files. {sum(1 for r in results if r['result'] is not None)} successful."

                return status_msg, table_data, zip_path

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                raise gr.Error(f"Batch processing failed: {str(e)}")

        # Wire up event handlers
        process_btn.click(
            fn=process_batch,
            inputs=[file_uploader, model_dropdown, language_dropdown],
            outputs=[status_text, results_table, batch_download]
        )

    return tab


def create_model_comparison_tab() -> gr.Tab:
    """
    Create the model comparison tab for benchmarking results.

    Returns:
        gr.Tab: Configured model comparison tab
    """
    with gr.TabItem("📊 Model Comparison", id="comparison") as tab:

        gr.Markdown("""
        # 📊 Model Performance Comparison
        Compare different speech-to-text models on Arabic audio datasets.
        """)

        with gr.Row():
            with gr.Column():
                # Controls
                dataset_dropdown = gr.Dropdown(
                    choices=["Common Voice Arabic", "Test Set"],
                    value="Test Set",
                    label="Dataset"
                )

                metrics_checkboxes = gr.CheckboxGroup(
                    choices=["WER", "CER", "Inference Time", "Memory Usage"],
                    value=["WER", "CER", "Inference Time"],
                    label="Metrics to Display"
                )

                refresh_btn = gr.Button("🔄 Refresh Results")

            with gr.Column():
                # Charts
                wer_chart = gr.Plot(label="Word Error Rate (WER)")
                time_chart = gr.Plot(label="Inference Time")

        # Results table
        benchmark_table = gr.Dataframe(
            headers=["Model", "WER", "CER", "Avg Time (s)", "Memory (MB)"],
            label="Detailed Benchmark Results"
        )

        # Download button
        download_results = gr.File(
            label="Download Raw Results",
            visible=False
        )

        # Event handlers
        def load_comparison_data(dataset: str, metrics: List[str]) -> Tuple[Any, Any, List[List], str]:
            """Load and display model comparison data."""
            try:
                # Load benchmark results
                results_file = Path(__file__).parent.parent / "results" / f"benchmark_{dataset.lower().replace(' ', '_')}.json"

                if results_file.exists():
                    with open(results_file, 'r', encoding='utf-8') as f:
                        benchmark_data = json.load(f)
                else:
                    # Generate sample data for demo
                    benchmark_data = {
                        "models": AVAILABLE_MODELS,
                        "results": {
                            model: {
                                "wer": 0.15 + i * 0.05,  # Sample WER values
                                "cer": 0.10 + i * 0.03,  # Sample CER values
                                "avg_time": 1.0 + i * 0.5,  # Sample times
                                "memory_mb": 1000 + i * 200  # Sample memory usage
                            }
                            for i, model in enumerate(AVAILABLE_MODELS)
                        }
                    }

                # Create WER chart
                if "WER" in metrics:
                    wer_data = {
                        "Model": list(benchmark_data["results"].keys()),
                        "WER": [data["wer"] for data in benchmark_data["results"].values()]
                    }
                    wer_fig = px.bar(
                        wer_data,
                        x="Model",
                        y="WER",
                        title="Word Error Rate by Model",
                        color="WER",
                        color_continuous_scale="RdYlGn_r"
                    )
                else:
                    wer_fig = None

                # Create time chart
                if "Inference Time" in metrics:
                    time_data = {
                        "Model": list(benchmark_data["results"].keys()),
                        "Time (s)": [data["avg_time"] for data in benchmark_data["results"].values()]
                    }
                    time_fig = px.bar(
                        time_data,
                        x="Model",
                        y="Time (s)",
                        title="Average Inference Time by Model",
                        color="Time (s)",
                        color_continuous_scale="Blues_r"
                    )
                else:
                    time_fig = None

                # Create table data
                table_data = [
                    [
                        model,
                        ".3f",
                        ".3f",
                        ".2f",
                        data["memory_mb"]
                    ]
                    for model, data in benchmark_data["results"].items()
                ]

                # Save raw results for download
                temp_file = tempfile.NamedTemporaryFile(
                    mode='w',
                    suffix='.json',
                    delete=False
                )
                json.dump(benchmark_data, temp_file, indent=2)
                temp_file.close()

                return wer_fig, time_fig, table_data, temp_file.name

            except Exception as e:
                logger.error(f"Failed to load comparison data: {e}")
                raise gr.Error(f"Failed to load comparison data: {str(e)}")

        # Wire up event handlers
        refresh_btn.click(
            fn=load_comparison_data,
            inputs=[dataset_dropdown, metrics_checkboxes],
            outputs=[wer_chart, time_chart, benchmark_table, download_results]
        )

        # Load initial data
        load_comparison_data("Test Set", ["WER", "CER", "Inference Time"])

    return tab


def create_about_tab() -> gr.Tab:
    """
    Create the about and help tab.

    Returns:
        gr.Tab: Configured about tab
    """
    with gr.TabItem("ℹ️ About & Help", id="about") as tab:

        gr.Markdown("""
        # ℹ️ About Arabic Speech-to-Text System

        A state-of-the-art Arabic speech recognition system built with modern deep learning techniques.
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                ## 🖥️ System Information

                **Version**: 1.0.0
                **Models**: Whisper (OpenAI)
                **Language**: Arabic (Primary), English (Secondary)
                **Framework**: PyTorch 2.0+
                """)

                # System info display
                system_info = gr.JSON(
                    label="Runtime Information",
                    value={
                        "cuda_available": config.get("gpu.enabled", False),
                        "pytorch_version": "2.0+",
                        "gradio_version": gr.__version__,
                        "supported_formats": ["wav", "mp3", "flac", "ogg", "m4a"]
                    }
                )

            with gr.Column():
                gr.Markdown("""
                ## 📚 References & Citations

                ### Core Technology
                - **OpenAI Whisper**: Radford et al. "Robust Speech Recognition via Large-Scale Weak Supervision"
                - **Hugging Face Transformers**: Wolf et al. "HuggingFace's Transformers: State-of-the-art Natural Language Processing"

                ### Dataset
                - **Mozilla Common Voice Arabic**: Open-source Arabic speech corpus

                ### Research Papers
                - Arabic Speech Recognition: State-of-the-art and future directions
                - Cross-lingual transfer learning for low-resource languages
                """)

                gr.Markdown("""
                ## 📞 Contact & Support

                - **Email**: support@arabic-stt.example.com
                - **GitHub Issues**: [Report bugs & request features](https://github.com/example/arabic-stt)
                - **Documentation**: [Full API docs](https://docs.arabic-stt.example.com)
                """)

    return tab


def create_gradio_interface() -> gr.Blocks:
    """
    Create the main Gradio interface with all tabs.

    Returns:
        gr.Blocks: Configured Gradio interface
    """
    # Custom CSS for Arabic/RTL support and styling
    css = """
    .gradio-container {
        direction: ltr;
    }
    .arabic-text {
        direction: rtl;
        font-family: 'Droid Sans Arabic', 'Arial Unicode MS', sans-serif;
    }
    .confidence-high {
        color: #10b981;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f59e0b;
    }
    .confidence-low {
        color: #ef4444;
    }
    """

    with gr.Blocks(title="Arabic Speech-to-Text System") as interface:

        gr.Markdown("""
        # 🎤 Arabic Speech-to-Text System
        Advanced AI-powered speech recognition for Arabic language with real-time transcription,
        batch processing, and model comparison capabilities.
        """)

        # Create tabs
        transcription_tab = create_transcription_tab()
        batch_tab = create_batch_processing_tab()
        comparison_tab = create_model_comparison_tab()
        about_tab = create_about_tab()

        # Footer
        gr.Markdown("""
        ---
        **Built with ❤️ using PyTorch, Whisper, and Gradio** |
        [View Source Code](https://github.com/example/arabic-stt) |
        [API Documentation](https://docs.arabic-stt.example.com)
        """)

    return interface


def main():
    """Main entry point for the Gradio application."""
    try:
        # Initialize services
        logger.info("Initializing Arabic Speech-to-Text Gradio interface...")
        logger.info(f"Demo service initialized: {demo_service}")
        logger.info(f"Batch processor initialized: {batch_processor}")

        # Create interface
        logger.info("Creating Gradio interface...")
        interface = create_gradio_interface()
        logger.info("Gradio interface created successfully")

        # Launch the interface
        logger.info("Launching Gradio server on 0.0.0.0:7860...")
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False
        )

        logger.info("Gradio interface launched successfully")

    except ImportError as e:
        logger.error(f"Import error: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"Failed to launch Gradio interface: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
