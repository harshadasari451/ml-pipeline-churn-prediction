"""
Logging configuration and utilities for the ML pipeline.
"""
import logging
import logging.config
import os
from pathlib import Path
from typing import Optional

import yaml


def setup_logging(
    config_path: str = "config/logging_config.yaml",
    default_level: int = logging.INFO
) -> None:
    """
    Setup logging configuration.
    
    Args:
        config_path: Path to the logging configuration file
        default_level: Default logging level if config file not found
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Load logging configuration
    path = Path(config_path)
    if path.exists():
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logging.warning(f"Logging config file not found at {config_path}, using basic config")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Name of the logger (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return logging.getLogger(name)
