import logging
import sys
from datetime import UTC, datetime  # Updated import
from pathlib import Path
from ..config import LOG_DIR


def setup_logging(
    log_dir: str = LOG_DIR, log_filename: str | None = None, level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging.

    Args:
        log_dir (str): Directory where the log files will be stored.
        log_filename (str | None): Name of the log file. If None, defaults to 'main_<YYYYMMDD>.log'.
        level (int): Logging level. Defaults to logging.INFO.

    Returns:
        logging.Logger: Configured logger instance.

    """
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True, parents=True)

    # Generate log filename if not provided
    if log_filename is None:
        log_filename = f"main_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = (
        False  # Prevent logs from being passed to the root logger multiple times
    )

    # Remove existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create handlers
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path / log_filename)

    # Set levels for handlers
    stream_handler.setLevel(level)
    file_handler.setLevel(level)

    # Create formatter and set it for handlers
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s:%(lineno)d] [%(funcName)s]: %(message)s"
    )
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
