import os
import sys
import logging
# from datetime import datetime
# from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

# Optional: colorized logs in terminal (requires 'colorama')
try:
    from colorama import Fore, Style, init
    init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

try:
    # works when imported as a module
    from ._configurations import configs
except (ImportError, SystemError, ValueError):
    # Fallback for running as __main__ (direct script execution)
    package_dir = (
        Path(os.path.abspath(__file__))
        .parent
        .parent  # go up one level to 'my_utilities' directory
    )
    sys.path.insert(0, str(package_dir))
    try:
        from _configurations import configs
    finally:
        sys.path.pop(0)

# initializations
# Global dictionary to hold loggers
_loggers = {}

# Log format
LOG_FORMAT = r'[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = r'%Y-%m-%d %H:%M:%S'

class ColoredFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.LIGHTBLACK_EX,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.CRITICAL: Fore.MAGENTA,
        logging.ERROR: Fore.RED,
    }

    def format(self, record):

        # Temporarily color only the level name
        current_levelname = record.levelname
        if COLOR_ENABLED:
            color = self.COLORS.get(record.levelno, '')
            record.levelname = f'{color}{record.levelname}{Style.RESET_ALL}'
        message = super().format(record)
        record.levelname = current_levelname

        return message


def setup_logger(name='Unknown', level=None, force=False) -> logging.Logger:
    global _loggers

    # Set default level from configs if not provided
    if level is None:
        level = configs.log.level

    # Check if logger already exists
    if name in _loggers and not force:
        return _loggers[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    _loggers[name] = logger

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(ch)

    return logger


if __name__ == '__main__':
    # an example of logger instance
    log = setup_logger(name='test_logger', level=logging.DEBUG)
    log.debug('Parsed production page.')
    log.info('utility started.')
    log.warning('We are seeing warnings!')
    log.critical('This is a critical event!')
    log.error('Failed to send the package, error!')