# rdkit_cli/core/config.py
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional


class RDKitCLIConfig:
    """Configuration manager for RDKit CLI."""
    
    def __init__(self) -> None:
        self.config_dir = Path.home() / ".config" / "rdkit-cli"
        self.config_file = self.config_dir / "config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self._config: Dict[str, Any] = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """Return default configuration."""
        return {
            "default_fps": "morgan",
            "default_radius": 2,
            "default_n_bits": 2048,
            "default_chunk_size": 1000,
            "default_jobs": os.cpu_count() or 1,
            "default_timeout": 300,
            "memory_limit": "8GB",
            "temp_dir": str(Path.home() / ".cache" / "rdkit-cli"),
        }
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self._config, f, indent=2)
        except IOError as e:
            import logging
            logger = logging.getLogger("rdkit_cli")
            logger.error(f"Failed to save configuration: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self._config[key] = value
        self.save()
    
    def list_all(self) -> Dict[str, Any]:
        """List all configuration values."""
        return self._config.copy()
    
    def reset(self) -> None:
        """Reset configuration to defaults."""
        self._config = self._default_config()
        self.save()


# Global config instance
config = RDKitCLIConfig()