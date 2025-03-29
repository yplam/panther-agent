import logging
import sys
from app import config

def setup_logger():
    """Configures the application logger."""
    logger = logging.getLogger("AI_Agent_Server")
    logger.setLevel(config.LOG_LEVEL)

    # Prevent duplicate handlers if called multiple times
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()