"""
Whisper Fine-Tuner for Arabic ASR (T068-T071)

Implements fine-tuning, validation, and checkpointing for Whisper models.
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import json
from dataclasses import dataclass, asdict
import time

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from jiwer import wer

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for fine-tuning Whisper."""
    
    model_size: str = "small"
    learning_rate: float = 1e-5
    batch_size: int = 16
    epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    seed: int = 42
    
    # Device configuration
    device: str = "auto"
    
    # Output
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    
    # Early stopping
    patience: int = 3
    eval_steps: int = 500
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Convert types from YAML strings
        converted = {}
        for key, value in config_dict.items():
            if key in cls.__dataclass_fields__:
                field_type = cls.__dataclass_fields__[key].type
                if field_type == int and isinstance(value, str):
                    converted[key] = int(value)
                elif field_type == float and isinstance(value, str):
                    converted[key] = float(value)
                elif field_type == bool and isinstance(value, str):
                    converted[key] = value.lower() in ('true', '1', 'yes', 'on')
                else:
                    converted[key] = value
        return cls(**converted)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    @staticmethod
    def resolve_device(device: Optional[str]) -> torch.device:
        """Resolve a device string to a torch.device."""
        device_str = (device or "auto").lower()
        if device_str == "cuda":
            if torch.cuda.is_available():
                return torch.device("cuda")
            logger.warning("CUDA requested but not available; falling back to CPU")
            return torch.device("cpu")
        if device_str in ("auto", "cpu"):
            if device_str == "auto" and torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        logger.warning(f"Invalid device '{device_str}' specified; falling back to CPU")
        return torch.device("cpu")


class WhisperFinetuner:
    """
    Fine-tunes OpenAI Whisper model on custom datasets.
    
    Handles:
    - Model loading and setup
    - Training loop with gradient accumulation
    - Validation loop with WER computation
    - Learning rate scheduling with warmup
    - Checkpoint saving/loading with best model selection
    - Early stopping
    """
    
    def __init__(
        self,
        config: TrainingConfig,
        processor: Optional[WhisperProcessor] = None,
        model: Optional[WhisperForConditionalGeneration] = None
    ):
        """
        Initialize the fine-tuner.
        
        Args:
            config: Training configuration
            processor: Whisper processor (will be loaded if not provided)
            model: Whisper model (will be loaded if not provided)
        """
        self.config = config
        self.device = TrainingConfig.resolve_device(config.device)
        
        # Set seed for reproducibility
        torch.manual_seed(config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(config.seed)
        
        # Create output directories
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # Load processor and model if not provided
        logger.info(f"Loading Whisper {config.model_size}")
        model_id = f"openai/whisper-{config.model_size}"
        
        if processor is None:
            self.processor = WhisperProcessor.from_pretrained(model_id, language="ar", task="transcribe")
        else:
            self.processor = processor
        
        if model is None:
            self.model = WhisperForConditionalGeneration.from_pretrained(model_id)
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Freeze encoder for efficient training
        self.model.model.encoder.requires_grad_(False)
        
        logger.info(f"Model loaded: {config.model_size}")
        logger.info(f"Encoder frozen, only decoder will be trained")
        
        # Training state
        self.best_wer = float('inf')
        self.best_model_path = None
        self.patience_counter = 0
        self.global_step = 0
        
    def prepare_optimizer(self, total_steps: int) -> Tuple[AdamW, torch.optim.lr_scheduler.LambdaLR]:
        """
        Prepare optimizer and learning rate scheduler.
        
        Args:
            total_steps: Total training steps for scheduler
            
        Returns:
            Tuple of (optimizer, scheduler)
        """
        # Only optimize decoder parameters
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "decoder" in n and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            }
        ]
        
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
        
        # Linear warmup then linear decay
        def lr_lambda(current_step: int) -> float:
            if current_step < self.config.warmup_steps:
                return float(current_step) / float(max(1, self.config.warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - self.config.warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        logger.info(f"Optimizer: AdamW (lr={self.config.learning_rate})")
        logger.info(f"Scheduler: Linear warmup {self.config.warmup_steps} steps then decay")
        
        return optimizer, scheduler
    
    def compute_wer(self, predictions: list, references: list) -> float:
        """
        Compute Word Error Rate (WER).
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            WER score (0.0 to 1.0)
        """
        try:
            return wer(references, predictions)
        except Exception as e:
            logger.error(f"Error computing WER: {e}")
            return 1.0
    
    def validate(self, val_dataloader: DataLoader) -> Tuple[float, float]:
        """
        Validate the model on a validation set.
        
        Args:
            val_dataloader: DataLoader for validation data
            
        Returns:
            Tuple of (validation_loss, WER)
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in val_dataloader:
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                text = batch["text"]
                
                # Forward pass
                outputs = self.model(
                    input_features=input_features,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Decode predictions
                predicted_ids = torch.argmax(outputs.logits, dim=-1)
                predictions = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                
                all_predictions.extend([p.lower() for p in predictions])
                all_references.extend([t.lower() for t in text])
        
        avg_loss = total_loss / len(val_dataloader)
        val_wer = self.compute_wer(all_predictions, all_references)
        
        return avg_loss, val_wer
    
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune the model.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data (optional)
            resume_from_checkpoint: Path to checkpoint to resume from (optional)
            
        Returns:
            Dictionary with training results and metrics
        """
        # Calculate total steps
        total_steps = len(train_dataloader) * self.config.epochs // self.config.gradient_accumulation_steps
        
        # Prepare optimizer and scheduler
        optimizer, scheduler = self.prepare_optimizer(total_steps)
        
        # Resume from checkpoint if provided
        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self._load_checkpoint(resume_from_checkpoint)
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        logger.info(f"Total steps: {total_steps}")
        
        training_results = {
            "epochs": self.config.epochs,
            "total_steps": total_steps,
            "train_losses": [],
            "val_losses": [],
            "val_wers": [],
            "best_wer": float('inf'),
            "best_checkpoint": None
        }
        
        # Training loop
        for epoch in range(self.config.epochs):
            self.model.train()
            epoch_loss = 0.0
            epoch_step = 0
            
            start_time = time.time()
            
            for batch_idx, batch in enumerate(train_dataloader):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_features=input_features,
                    labels=labels
                )
                
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                epoch_loss += loss.item()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    epoch_step += 1
                    self.global_step += 1
                    
                    if self.global_step % 100 == 0:
                        elapsed = time.time() - start_time
                        logger.info(
                            f"Epoch {epoch+1}/{self.config.epochs} | "
                            f"Step {epoch_step} | "
                            f"Loss: {epoch_loss / epoch_step:.4f} | "
                            f"Time: {elapsed:.1f}s | "
                            f"LR: {scheduler.get_last_lr()[0]:.2e}"
                        )
                    
                    # Validation during training
                    if val_dataloader and self.global_step % self.config.eval_steps == 0:
                        val_loss, val_wer = self.validate(val_dataloader)
                        logger.info(f"Validation: Loss={val_loss:.4f}, WER={val_wer:.4f}")
                        
                        training_results["val_losses"].append(val_loss)
                        training_results["val_wers"].append(val_wer)
                        
                        # Save best model
                        if val_wer < self.best_wer:
                            self.best_wer = val_wer
                            self.patience_counter = 0
                            self._save_checkpoint(epoch, epoch_step, val_wer)
                            training_results["best_wer"] = val_wer
                            training_results["best_checkpoint"] = self.best_model_path
                        else:
                            self.patience_counter += 1
                            
                            # Early stopping
                            if self.patience_counter >= self.config.patience:
                                logger.info(f"Early stopping triggered after {self.config.patience} validation steps without improvement")
                                training_results["early_stopped"] = True
                                return training_results
            
            avg_epoch_loss = epoch_loss / epoch_step if epoch_step > 0 else epoch_loss
            training_results["train_losses"].append(avg_epoch_loss)
            
            logger.info(f"Epoch {epoch+1} completed - Average Loss: {avg_epoch_loss:.4f}")
            
            # Final validation at end of epoch
            if val_dataloader:
                val_loss, val_wer = self.validate(val_dataloader)
                logger.info(f"End-of-epoch validation: Loss={val_loss:.4f}, WER={val_wer:.4f}")
                training_results["val_losses"].append(val_loss)
                training_results["val_wers"].append(val_wer)
                
                if val_wer < self.best_wer:
                    self.best_wer = val_wer
                    self.patience_counter = 0
                    self._save_checkpoint(epoch+1, epoch_step, val_wer)
                    training_results["best_wer"] = val_wer
                    training_results["best_checkpoint"] = self.best_model_path
        
        logger.info(f"Training completed. Best WER: {self.best_wer:.4f}")
        
        return training_results
    
    def _save_checkpoint(self, epoch: int, step: int, metric: float) -> None:
        """Save model checkpoint."""
        checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint-epoch{epoch+1}-wer{metric:.4f}"
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and processor
        self.model.save_pretrained(str(checkpoint_path))
        self.processor.save_pretrained(str(checkpoint_path))
        
        # Save training metadata
        metadata = {
            "epoch": epoch,
            "step": step,
            "metric": metric,
            "config": self.config.to_dict()
        }
        
        with open(checkpoint_path / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        self.best_model_path = str(checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model from checkpoint."""
        checkpoint_path = Path(checkpoint_path)
        
        self.model = WhisperForConditionalGeneration.from_pretrained(str(checkpoint_path))
        self.processor = WhisperProcessor.from_pretrained(str(checkpoint_path))
        self.model.to(self.device)
        
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
