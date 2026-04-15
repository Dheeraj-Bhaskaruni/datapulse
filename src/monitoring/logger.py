"""Structured logging setup for the entire pipeline."""

import logging
import logging.config
import json
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if record.exc_info and record.exc_info[0]:
            log_data['exception'] = self.formatException(record.exc_info)
        if hasattr(record, 'extra_data'):
            log_data['extra'] = record.extra_data
        return json.dumps(log_data)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None, json_format: bool = False) -> None:
    """Configure application logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    if json_format:
        formatter: logging.Formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for lib in ['urllib3', 'matplotlib', 'PIL']:
        logging.getLogger(lib).setLevel(logging.WARNING)

    logging.info(f"Logging configured: level={level}, file={log_file}")
