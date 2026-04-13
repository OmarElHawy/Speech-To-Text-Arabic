"""CLI commands for Speech to Text system (T038-T041)"""

import click
import sys
from pathlib import Path
from typing import Optional

from src.models.whisper_base import WhisperBaseModel
from src.services.transcription_pipeline import TranscriptionPipeline
from src.services.storage_service import StorageService
from src.utils.logging import get_logger, setup_simple_logging

logger = get_logger(__name__)


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--log-level', default='INFO', help='Logging level')
def cli(verbose: bool, log_level: str):
    """Speech to Text Arabic ASR System"""
    if verbose:
        log_level = 'DEBUG'
    
    setup_simple_logging(log_level)
    
    if verbose:
        logger.info("Verbose logging enabled")


@cli.command()
@click.argument('audio_path', type=click.Path(exists=True))
@click.option('--model', '-m', default='whisper-small', help='Model to use')
@click.option('--language', '-l', default='ar', help='Language code')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'txt', 'csv']), default='json', help='Output format')
@click.option('--no-timestamps', is_flag=True, help='Disable timestamp generation')
def transcribe(
    audio_path: str,
    model: str,
    language: str,
    output: Optional[str],
    format: str,
    no_timestamps: bool
):
    """
    Transcribe audio file to text
    
    AUDIO_PATH: Path to audio file to transcribe
    """
    try:
        # Parse model specification
        if model.startswith('whisper-'):
            model_size = model.split('-')[1]
            transcription_service = WhisperBaseModel(model_size=model_size)
        else:
            raise click.BadParameter(f"Unsupported model: {model}")
        
        # Load model
        click.echo(f"Loading {model} model...")
        transcription_service.load_model()
        
        # Create pipeline
        pipeline = TranscriptionPipeline(transcription_service)
        
        # Transcribe
        click.echo(f"Transcribing {audio_path}...")
        result = pipeline.transcribe_file(
            audio_path,
            language=language,
            generate_segments=not no_timestamps
        )
        
        # Display result
        click.echo("\nTranscription Result:")
        click.echo("=" * 50)
        click.echo(result.text)
        click.echo("=" * 50)
        
        if result.confidence_score is not None:
            click.echo(f"Confidence: {result.confidence_score:.3f}")
        
        if result.word_error_rate is not None:
            click.echo(f"Word Error Rate: {result.word_error_rate:.3f}")
        
        if result.processing_time_ms is not None:
            click.echo(f"Processing time: {result.processing_time_ms}ms")
        
        if result.segments and len(result.segments) > 0:
            click.echo(f"Segments: {len(result.segments)}")
        
        # Save result if requested
        if output:
            storage = StorageService()
            saved_path = storage.save_transcription_result(
                result, audio_path, model, format, output_path=output
            )
            click.echo(f"Result saved to: {saved_path}")
        
        # Cleanup
        transcription_service.unload_model()
        
    except Exception as e:
        logger.error(f"Transcription failed: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('audio_paths', nargs=-1, type=click.Path(exists=True))
@click.option('--model', '-m', default='whisper-small', help='Model to use')
@click.option('--language', '-l', default='ar', help='Language code')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory')
@click.option('--format', '-f', type=click.Choice(['json', 'csv']), default='json', help='Output format')
@click.option('--batch-name', help='Name for batch results')
def transcribe_batch(
    audio_paths: tuple,
    model: str,
    language: str,
    output_dir: Optional[str],
    format: str,
    batch_name: Optional[str]
):
    """
    Transcribe multiple audio files
    
    AUDIO_PATHS: Paths to audio files to transcribe
    """
    if not audio_paths:
        click.echo("Error: No audio files provided", err=True)
        sys.exit(1)
    
    try:
        # Parse model specification
        if model.startswith('whisper-'):
            model_size = model.split('-')[1]
            transcription_service = WhisperBaseModel(model_size=model_size)
        else:
            raise click.BadParameter(f"Unsupported model: {model}")
        
        # Load model
        click.echo(f"Loading {model} model...")
        transcription_service.load_model()
        
        # Create pipeline
        pipeline = TranscriptionPipeline(transcription_service)
        
        # Transcribe batch
        audio_list = list(audio_paths)
        click.echo(f"Transcribing {len(audio_list)} files...")
        
        results = pipeline.transcribe_batch(audio_list, language=language)
        
        # Display summary
        successful = sum(1 for r in results if r.text)
        total_time = sum(r.processing_time_ms or 0 for r in results)
        
        click.echo("\nBatch Transcription Summary:")
        click.echo("=" * 50)
        click.echo(f"Total files: {len(results)}")
        click.echo(f"Successful: {successful}")
        click.echo(f"Failed: {len(results) - successful}")
        click.echo(f"Total processing time: {total_time}ms")
        if len(results) > 0:
            click.echo(f"Average time per file: {total_time / len(results):.1f}ms")
        
        # Save batch results if requested
        if output_dir:
            storage = StorageService(output_dir)
            saved_path = storage.save_batch_results(
                results, audio_list, model, format, batch_name
            )
            click.echo(f"Batch results saved to: {saved_path}")
        
        # Cleanup
        transcription_service.unload_model()
        
    except Exception as e:
        logger.error(f"Batch transcription failed: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def list_models():
    """List available models"""
    models = {
        'whisper': ['tiny', 'base', 'small', 'medium', 'large'],
        'wav2vec': ['xlsr-53-arabic'],
        'deepspeech': ['Not implemented yet']
    }
    
    click.echo("Available Models:")
    click.echo("=" * 50)
    
    for model_type, sizes in models.items():
        click.echo(f"{model_type.upper()}:")
        for size in sizes:
            click.echo(f"  - {model_type}-{size}")
        click.echo()


@cli.command()
@click.argument('model_name')
def download_model(model_name: str):
    """
    Download and verify a model
    
    MODEL_NAME: Name of model to download (e.g., whisper-small)
    """
    try:
        from scripts.download_models import ModelDownloader
        
        downloader = ModelDownloader()
        
        if model_name.startswith('whisper-'):
            model_size = model_name.split('-')[1]
            success = downloader.download_whisper_model(model_size)
        elif model_name.startswith('wav2vec-'):
            # Map common names to full model names
            model_map = {
                'xlsr-53-arabic': 'facebook/wav2vec2-xlsr-53-arabic'
            }
            full_name = model_map.get(model_name.split('-', 1)[1])
            if full_name:
                success = downloader.download_wav2vec_model(full_name)
            else:
                success = False
        else:
            click.echo(f"Unknown model: {model_name}", err=True)
            sys.exit(1)
        
        if success:
            click.echo(f"Successfully downloaded {model_name}")
        else:
            click.echo(f"Failed to download {model_name}", err=True)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Download failed: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind the server to')
@click.option('--port', '-p', default=7860, type=int, help='Port to run the server on')
@click.option('--model', '-m', default='whisper-small', help='Default model for demo')
@click.option('--share', is_flag=True, help='Create a public link')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def demo(host: str, port: int, model: str, share: bool, debug: bool):
    """
    Launch the Gradio web demo interface

    This command starts a web interface for the speech-to-text system
    with transcription, batch processing, and model comparison features.
    """
    try:
        import gradio as gr
        from demo.app_gradio import create_gradio_interface

        click.echo(f"Starting Arabic Speech-to-Text Demo...")
        click.echo(f"Host: {host}")
        click.echo(f"Port: {port}")
        click.echo(f"Default Model: {model}")
        click.echo(f"Share: {share}")
        click.echo(f"Debug: {debug}")
        click.echo("")

        # Set environment variables for the demo
        import os
        os.environ['GRADIO_SERVER_NAME'] = host
        os.environ['GRADIO_SERVER_PORT'] = str(port)

        if debug:
            os.environ['GRADIO_DEBUG'] = 'True'

        # Create and launch the interface
        interface = create_gradio_interface()

        click.echo("Launching Gradio interface...")
        click.echo(f"Demo will be available at: http://{host}:{port}")
        if share:
            click.echo("Public link will be generated...")

        # Launch with configuration
        interface.launch(
            server_name=host,
            server_port=port,
            share=share
        )

    except ImportError as e:
        click.echo(f"Error: Required packages not installed. {str(e)}", err=True)
        click.echo("Install demo dependencies: pip install gradio plotly", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to launch demo: {str(e)}")
        click.echo(f"Error launching demo: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', type=click.Path(exists=True), default='config/training_config.yaml', help='Training configuration YAML file')
@click.option('--data-dir', '-d', type=click.Path(exists=True), default=None, help='Path to Common Voice dataset')
@click.option('--model-size', '-m', type=click.Choice(['tiny', 'base', 'small', 'medium', 'large']), default=None, help='Whisper model size')
@click.option('--epochs', '-e', type=int, default=None, help='Number of training epochs')
@click.option('--batch-size', '-b', type=int, default=None, help='Batch size for training')
@click.option('--learning-rate', '-lr', type=float, default=None, help='Learning rate')
@click.option('--output-dir', '-o', type=click.Path(), default=None, help='Output directory for results')
@click.option('--checkpoint-dir', type=click.Path(), default=None, help='Directory for saving checkpoints')
@click.option('--resume', type=click.Path(exists=True), default=None, help='Path to checkpoint to resume from')
def train(config: str, data_dir: Optional[str], model_size: Optional[str], epochs: Optional[int], 
          batch_size: Optional[int], learning_rate: Optional[float], output_dir: Optional[str], 
          checkpoint_dir: Optional[str], resume: Optional[str]):
    """
    Fine-tune Whisper on Arabic Common Voice dataset (T075-T082)
    
    Example usage:
        python -m src.cli.commands train --config config/training_config.yaml
        python -m src.cli.commands train --model-size small --epochs 3 --batch-size 16
    """
    try:
        import subprocess
        
        click.echo("=" * 80)
        click.echo("WHISPER FINE-TUNING FOR ARABIC")
        click.echo("=" * 80)
        
        # Build command
        cmd = [sys.executable, "scripts/train_whisper.py"]
        
        if config:
            cmd.extend(["--config", config])
        if data_dir:
            cmd.extend(["--data-dir", data_dir])
        if model_size:
            cmd.extend(["--model-size", model_size])
        if epochs:
            cmd.extend(["--epochs", str(epochs)])
        if batch_size:
            cmd.extend(["--batch-size", str(batch_size)])
        if learning_rate:
            cmd.extend(["--learning-rate", str(learning_rate)])
        if output_dir:
            cmd.extend(["--output-dir", output_dir])
        if checkpoint_dir:
            cmd.extend(["--checkpoint-dir", checkpoint_dir])
        if resume:
            cmd.extend(["--resume", resume])
        
        click.echo(f"Running: {' '.join(cmd)}")
        click.echo("")
        
        # Execute training script
        result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent.parent))
        
        if result.returncode != 0:
            sys.exit(result.returncode)
            
    except Exception as e:
        logger.error(f"Failed to start training: {str(e)}")
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()
