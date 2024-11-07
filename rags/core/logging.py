import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path


def setup_logging(log_file_path: Path) -> None:
    """Configure logging to both console and file with daily rotation.

    Args:
        log_file_path: Path to the log file.
    """
    try:
        # Create logs directory if it doesn't exist
        log_dir_path = log_file_path.parent
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_path = log_dir_path / log_file_path

        # Configure root logger
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        # Detailed formatter
        formatter = logging.Formatter(
            fmt="%(asctime)s.%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation
        file_handler = TimedRotatingFileHandler(
            str(log_path),  # TimedRotatingFileHandler requires string path
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    except Exception as e:
        logging.error(f"Failed to setup logging: {str(e)}")
        raise
