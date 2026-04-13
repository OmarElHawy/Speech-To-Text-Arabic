#!/usr/bin/env python3
"""Model download and verification script (T037)

Downloads and verifies ASR models:
- OpenAI Whisper models
- Wav2Vec 2.0 models
- DeepSpeech models
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse

from src.utils.logging import get_logger

logger = get_logger(__name__)


class ModelDownloader:
    """Handles downloading and verification of ASR models"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model downloader
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        if cache_dir is None:
            # Use default HuggingFace cache
            import tempfile
            self.cache_dir = os.path.join(tempfile.gettempdir(), "speech_to_text_models")
        else:
            self.cache_dir = cache_dir
        
        Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Initialized ModelDownloader with cache_dir={self.cache_dir}")
    
    def download_whisper_model(self, model_size: str = "small") -> bool:
        """
        Download and verify Whisper model
        
        Args:
            model_size: Model size ('tiny', 'base', 'small', 'medium', 'large')
            
        Returns:
            True if successful
        """
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            
            model_name = f"openai/whisper-{model_size}"
            logger.info(f"Downloading Whisper model: {model_name}")
            
            # Download processor
            processor = WhisperProcessor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Download model
            model = WhisperForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Whisper {model_size}: {str(e)}")
            return False
    
    def download_wav2vec_model(self, model_name: str = "facebook/wav2vec2-xlsr-53-arabic") -> bool:
        """
        Download and verify Wav2Vec 2.0 model
        
        Args:
            model_name: HuggingFace model name
            
        Returns:
            True if successful
        """
        try:
            from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
            
            logger.info(f"Downloading Wav2Vec model: {model_name}")
            
            # Download processor
            processor = Wav2Vec2Processor.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            # Download model
            model = Wav2Vec2ForCTC.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            logger.info(f"Successfully downloaded {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to download Wav2Vec {model_name}: {str(e)}")
            return False
    
    def download_deepspeech_model(self) -> bool:
        """
        Download and verify DeepSpeech model
        
        Note: DeepSpeech models are typically hosted separately from HuggingFace
        This is a placeholder for future implementation
        
        Returns:
            True if successful
        """
        logger.warning("DeepSpeech model download not implemented yet")
        logger.info("DeepSpeech models need to be downloaded manually from:")
        logger.info("https://github.com/mozilla/DeepSpeech/releases")
        return False
    
    def verify_model(self, model_type: str, model_name: str) -> bool:
        """
        Verify that a model is properly downloaded and functional
        
        Args:
            model_type: Type of model ('whisper', 'wav2vec', 'deepspeech')
            model_name: Model name or size
            
        Returns:
            True if model is verified
        """
        try:
            if model_type == "whisper":
                from transformers import WhisperProcessor, WhisperForConditionalGeneration
                
                processor = WhisperProcessor.from_pretrained(
                    f"openai/whisper-{model_name}",
                    cache_dir=self.cache_dir
                )
                model = WhisperForConditionalGeneration.from_pretrained(
                    f"openai/whisper-{model_name}",
                    cache_dir=self.cache_dir
                )
                
            elif model_type == "wav2vec":
                from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
                
                processor = Wav2Vec2Processor.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
                model = Wav2Vec2ForCTC.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
            
            else:
                logger.error(f"Unknown model type: {model_type}")
                return False
            
            logger.info(f"Successfully verified {model_type} model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to verify {model_type} {model_name}: {str(e)}")
            return False
    
    def list_downloaded_models(self) -> Dict[str, List[str]]:
        """
        List downloaded models in cache directory
        
        Returns:
            Dictionary mapping model types to lists of model names
        """
        models = {
            "whisper": [],
            "wav2vec": [],
            "deepspeech": []
        }
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            # Check for Whisper models
            for size in ["tiny", "base", "small", "medium", "large"]:
                model_name = f"openai/whisper-{size}"
                try:
                    api.model_info(model_name)
                    if self.verify_model("whisper", size):
                        models["whisper"].append(size)
                except:
                    pass
            
            # Check for Wav2Vec models
            wav2vec_models = [
                "facebook/wav2vec2-xlsr-53-arabic",
                "facebook/wav2vec2-large-xlsr-53-arabic"
            ]
            
            for model_name in wav2vec_models:
                try:
                    api.model_info(model_name)
                    if self.verify_model("wav2vec", model_name):
                        models["wav2vec"].append(model_name.split("/")[-1])
                except:
                    pass
            
        except ImportError:
            logger.warning("huggingface_hub not available for model listing")
        
        return models
    
    def get_cache_size(self) -> Dict[str, float]:
        """
        Get cache directory size information
        
        Returns:
            Dictionary with size information
        """
        total_size = 0
        
        for file_path in Path(self.cache_dir).rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return {
            "cache_dir": self.cache_dir,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024)
        }


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description="Download and verify ASR models"
    )
    parser.add_argument(
        "--model-type",
        choices=["whisper", "wav2vec", "deepspeech", "all"],
        default="all",
        help="Type of model to download"
    )
    parser.add_argument(
        "--whisper-size",
        choices=["tiny", "base", "small", "medium", "large"],
        default="small",
        help="Whisper model size"
    )
    parser.add_argument(
        "--wav2vec-model",
        default="facebook/wav2vec2-xlsr-53-arabic",
        help="Wav2Vec model name"
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory for models"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models, don't download"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List downloaded models"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.cache_dir)
    
    if args.list_models:
        models = downloader.list_downloaded_models()
        cache_info = downloader.get_cache_size()
        
        print("Downloaded Models:")
        for model_type, model_list in models.items():
            print(f"  {model_type}: {model_list}")
        
        print(f"\nCache Info: {cache_info}")
        return
    
    success_count = 0
    total_count = 0
    
    # Download Whisper models
    if args.model_type in ["whisper", "all"]:
        if not args.verify_only:
            logger.info(f"Downloading Whisper {args.whisper_size}...")
            if downloader.download_whisper_model(args.whisper_size):
                success_count += 1
            total_count += 1
        else:
            logger.info(f"Verifying Whisper {args.whisper_size}...")
            if downloader.verify_model("whisper", args.whisper_size):
                success_count += 1
            total_count += 1
    
    # Download Wav2Vec models
    if args.model_type in ["wav2vec", "all"]:
        if not args.verify_only:
            logger.info(f"Downloading Wav2Vec {args.wav2vec_model}...")
            if downloader.download_wav2vec_model(args.wav2vec_model):
                success_count += 1
            total_count += 1
        else:
            logger.info(f"Verifying Wav2Vec {args.wav2vec_model}...")
            if downloader.verify_model("wav2vec", args.wav2vec_model):
                success_count += 1
            total_count += 1
    
    # Download DeepSpeech models
    if args.model_type in ["deepspeech", "all"]:
        if not args.verify_only:
            logger.info("Downloading DeepSpeech...")
            if downloader.download_deepspeech_model():
                success_count += 1
            total_count += 1
    
    logger.info(f"Download complete: {success_count}/{total_count} successful")
    
    if success_count != total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
