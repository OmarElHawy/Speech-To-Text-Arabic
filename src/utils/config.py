"""Configuration management utilities (T016)"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manager for loading and accessing configuration from YAML and environment"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager
        
        Args:
            config_path: Optional path to YAML config file
        """
        self.config = {}
        
        if config_path:
            self.load_yaml(config_path)
        else:
            # Try to load default config if it exists
            default_path = Path(__file__).parent.parent / "config" / "config.yaml"
            if default_path.exists():
                self.load_yaml(str(default_path))

    def load_yaml(self, path: str) -> None:
        """Load configuration from YAML file"""
        config_path = Path(path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                loaded_config = yaml.safe_load(f) or {}
            self.config.update(loaded_config)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {str(e)}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with dot notation support
        
        Args:
            key: Configuration key (supports dot notation: "section.subsection.key")
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
        
        # Check environment variable override
        env_key = key.upper().replace('.', '_')
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Try to convert string to appropriate type
            if env_value.lower() in ('true', 'false'):
                return env_value.lower() == 'true'
            try:
                return float(env_value) if '.' in env_value else int(env_value)
            except ValueError:
                return env_value
        
        return value

    def set(self, key: str, value: Any) -> None:
        """Set configuration value with dot notation support"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        logger.debug(f"Set config {key} = {value}")

    def get_section(self, section: str, default: Dict = None) -> Dict:
        """Get entire config section"""
        return self.config.get(section, default or {})

    def to_dict(self) -> Dict:
        """Export configuration as dictionary"""
        return self.config.copy()

    def __repr__(self):
        return f"ConfigManager({self.config})"


# Global config instance
_global_config = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create global config manager"""
    global _global_config
    if _global_config is None:
        _global_config = ConfigManager(config_path)
    return _global_config


def reset_config() -> None:
    """Reset global config (useful for testing)"""
    global _global_config
    _global_config = None


# Alias for backward compatibility
Config = ConfigManager
