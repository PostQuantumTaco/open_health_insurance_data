"""
Configuration loader for YAML config files.
Provides centralized access to runtime and distribution parameters.
"""

import yaml
import os
import multiprocessing
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Loads and provides access to configuration parameters."""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the config loader.

        Args:
            config_path: Path to main config.yaml file
        """
        self.config_path = Path(config_path)
        self.project_root = self._find_project_root()

        # Load main configuration
        self.config = self._load_yaml(self.project_root / config_path)

        # Load distributions configuration
        distributions_path = self.project_root / "config" / self.config.get('distributions_file', 'distributions.yaml')
        self.distributions = self._load_yaml(distributions_path)

    def _find_project_root(self) -> Path:
        """Find the project root directory (contains config/ folder)."""
        current = Path.cwd()

        # Check if we're already in the project root
        if (current / "config").exists():
            return current

        # Check parent directories
        for parent in current.parents:
            if (parent / "config").exists():
                return parent

        # If not found, assume current directory
        return current

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Load a YAML file.

        Args:
            path: Path to YAML file

        Returns:
            Dictionary of configuration parameters
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value from main config.

        Args:
            key: Configuration key (supports dot notation, e.g., 'null_rates.heart_rate')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        return self._get_nested(self.config, key, default)

    def get_distribution(self, key: str, default: Any = None) -> Any:
        """
        Get a distribution parameter.

        Args:
            key: Distribution key (supports dot notation)
            default: Default value if key not found

        Returns:
            Distribution parameter value
        """
        return self._get_nested(self.distributions, key, default)

    def _get_nested(self, d: Dict, key: str, default: Any = None) -> Any:
        """
        Get a nested dictionary value using dot notation.

        Args:
            d: Dictionary to search
            key: Key with dot notation (e.g., 'parent.child.grandchild')
            default: Default value if key not found

        Returns:
            Value at the key path
        """
        keys = key.split('.')
        current = d

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current

    def get_reference_data_path(self, filename: str) -> Path:
        """
        Get full path to a reference data file.

        Args:
            filename: Name of the reference data file

        Returns:
            Full path to the file
        """
        ref_dir = self.config.get('reference_data_dir', 'config/reference_data')
        return self.project_root / ref_dir / filename

    def get_output_dir(self) -> Path:
        """
        Get the output directory path, creating it if it doesn't exist.

        Returns:
            Path to output directory
        """
        output_dir = self.project_root / self.config.get('output_dir', 'synthetic_data')
        output_dir.mkdir(exist_ok=True)
        return output_dir

    def get_n_workers(self) -> int:
        """
        Get the number of workers for parallel processing.
        If n_workers is null, auto-detect based on CPU count.

        Returns:
            Number of workers to use
        """
        n_workers = self.config.get('n_workers')
        if n_workers is None:
            # Auto-detect: use CPU count - 1 (minimum 1)
            n_workers = max(1, multiprocessing.cpu_count() - 1)
        return int(n_workers)

    def validate(self) -> None:
        """
        Validate configuration parameters.
        Raises ValueError if configuration is invalid.
        """
        # Validate n_members
        n_members = self.config.get('n_members')
        if n_members is None or n_members <= 0:
            raise ValueError(f"n_members must be positive, got {n_members}")

        # Validate chunk_size
        chunk_size = self.config.get('chunk_size')
        if chunk_size is not None and chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        # Validate n_workers
        n_workers = self.config.get('n_workers')
        if n_workers is not None and n_workers <= 0:
            raise ValueError(f"n_workers must be positive, got {n_workers}")

        # Validate enable_parallel
        enable_parallel = self.config.get('enable_parallel', False)
        if not isinstance(enable_parallel, bool):
            raise ValueError(f"enable_parallel must be boolean, got {type(enable_parallel)}")

        # Warn if chunk_size > n_members
        if chunk_size is not None and chunk_size > n_members:
            print(f"Warning: chunk_size ({chunk_size}) > n_members ({n_members}). Setting chunk_size = n_members.")
            self.config['chunk_size'] = n_members

    def __repr__(self) -> str:
        """String representation of the config loader."""
        parallel_info = ""
        if self.config.get('enable_parallel', False):
            parallel_info = f", parallel={self.get_n_workers()} workers"
        return f"ConfigLoader(n_members={self.get('n_members')}, seed={self.get('random_seed')}{parallel_info})"


# Global config instance (singleton pattern)
_config_instance = None


def get_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Get the global configuration instance.

    Args:
        config_path: Path to config file (only used on first call)

    Returns:
        Global ConfigLoader instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigLoader(config_path)
    return _config_instance


def reload_config(config_path: str = "config/config.yaml") -> ConfigLoader:
    """
    Force reload the configuration.

    Args:
        config_path: Path to config file

    Returns:
        Newly loaded ConfigLoader instance
    """
    global _config_instance
    _config_instance = ConfigLoader(config_path)
    return _config_instance
