# rdkit_cli/core/logging.py
import logging
import sys
from datetime import datetime
from typing import Optional

import colorlog


class RDKitCLIFormatter(colorlog.ColoredFormatter):
    """Custom formatter for RDKit CLI with structured time-based output."""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%H:%M")
        record.timestamp = timestamp
        return super().format(record)


def setup_logging(verbose: bool = False, debug: bool = False) -> logging.Logger:
    """Setup colorized logging with structured output format.
    
    Args:
        verbose: Enable INFO level logging
        debug: Enable DEBUG level logging
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("rdkit_cli")
    logger.handlers.clear()
    
    if debug:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING
    
    logger.setLevel(level)
    
    handler = colorlog.StreamHandler(sys.stderr)
    handler.setLevel(level)
    
    formatter = RDKitCLIFormatter(
        '%(timestamp)s - %(log_color)s%(levelname)s%(reset)s : %(message)s',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'blue',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger


def log_success(message: str) -> None:
    """Log a success message with green color."""
    logger = logging.getLogger("rdkit_cli")
    if logger.isEnabledFor(logging.INFO):
        timestamp = datetime.now().strftime("%H:%M")
        print(f"{timestamp} - \033[92mSUCCESS\033[0m : {message}", file=sys.stderr)