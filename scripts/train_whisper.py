#!/usr/bin/env python3
"""
Training Script for Fine-tuning Whisper on Arabic Data (T075-T078)

Usage:
    python scripts/train_whisper.py --config config/training_config.yaml
    python scripts/train_whisper.py --model-size small --epochs 3 --batch-size 16
"""

import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from src.models.whisper_finetuner import WhisperFinetuner, TrainingConfig
from src.services.common_voice_dataset import CommonVoiceDataset
from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_dataloaders(
    data_dir: str,
    processor: WhisperProcessor,
    config: TrainingConfig,
    training_config: dict
) -> tuple:
    """Create train and validation dataloaders."""
    dataset_config = training_config.get("dataset", {})
    
    logger.info("Creating training dataset")
    train_dataset = CommonVoiceDataset(
        data_dir=data_dir,
        split="train",
        processor=processor,
        max_duration=dataset_config.get("max_duration", 30.0),
        sample_rate=dataset_config.get("sample_rate", 16000),
        augment=dataset_config.get("augment", True)
    )
    
    logger.info("Creating validation dataset")
    val_dataset = CommonVoiceDataset(
        data_dir=data_dir,
        split="dev",
        processor=processor,
        max_duration=dataset_config.get("max_duration", 30.0),
        sample_rate=dataset_config.get("sample_rate", 16000),
        augment=False  # No augmentation for validation
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Pad to the longest sequence in the batch
    input_features = [item["input_features"] for item in batch]
    labels = [item["labels"] for item in batch]
    texts = [item["text"] for item in batch]
    
    # Stack input features
    input_features_batch = torch.stack([torch.squeeze(f, 0) for f in input_features])
    
    # Pad labels
    max_label_length = max(len(l) for l in labels)
    labels_batch = torch.full((len(labels), max_label_length), -100, dtype=torch.long)
    for i, label in enumerate(labels):
        labels_batch[i, :len(label)] = torch.tensor(label)
    
    return {
        "input_features": input_features_batch,
        "labels": labels_batch,
        "text": texts
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Fine-tune Whisper on Arabic data")
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training configuration YAML file"
    )
    
    # Model configuration
    parser.add_argument(
        "--model-size",
        type=str,
        default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (overrides config)"
    )
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda", "auto"], help="Device to use for training")
    
    # Dataset
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to Common Voice dataset directory"
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for saving checkpoints"
    )
    
    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config_dict = load_config(args.config)
    
    # Override with command line arguments
    if args.model_size:
        config_dict["model_size"] = args.model_size
    if args.epochs:
        config_dict["epochs"] = args.epochs
    if args.batch_size:
        config_dict["batch_size"] = args.batch_size
    if args.learning_rate:
        config_dict["learning_rate"] = args.learning_rate
    if args.output_dir:
        config_dict["output_dir"] = args.output_dir
    if args.checkpoint_dir:
        config_dict["checkpoint_dir"] = args.checkpoint_dir
    if args.device:
        config_dict["device"] = args.device
    
    # Create training config
    config = TrainingConfig.from_dict(config_dict)
    
    logger.info(f"Training Configuration:")
    logger.info(f"  Model: Whisper {config.model_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  Output dir: {config.output_dir}")
    
    # Determine data directory
    data_dir = args.data_dir or config_dict.get("dataset", {}).get("data_dir", "./cv-corpus-24.0-2025-12-05/ar")
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        logger.error(f"Dataset directory not found: {data_dir}")
        return 1
    
    logger.info(f"Dataset directory: {data_dir}")
    
    # Initialize processor and model
    logger.info(f"Loading Whisper {config.model_size}")
    model_id = f"openai/whisper-{config.model_size}"
    processor = WhisperProcessor.from_pretrained(model_id, language="ar", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    
    # Create fine-tuner
    finetuner = WhisperFinetuner(config, processor=processor, model=model)
    
    # Create dataloaders
    logger.info("Preparing datasets")
    train_dataloader, val_dataloader = create_dataloaders(
        str(data_dir),
        processor,
        config,
        config_dict
    )
    
    # Start training
    logger.info("=" * 80)
    logger.info("STARTING FINE-TUNING")
    logger.info("=" * 80)
    
    start_time = time.time()
    
    try:
        training_results = finetuner.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            resume_from_checkpoint=args.resume
        )
        
        elapsed_time = time.time() - start_time
        
        # Save training results
        results_file = Path(config.output_dir) / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        training_results["elapsed_time_seconds"] = elapsed_time
        training_results["elapsed_time_hours"] = elapsed_time / 3600
        
        with open(results_file, "w") as f:
            json.dump(training_results, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Total time: {elapsed_time/3600:.2f} hours")
        logger.info(f"Best WER: {finetuner.best_wer:.4f}")
        logger.info(f"Best checkpoint: {finetuner.best_model_path}")
        logger.info(f"Results saved to: {results_file}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
