import os
import sys
import logging
import re
import threading
import time
import json
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
    if COLOR_ENABLED:
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


class RestrictedLogger(logging.Logger):

    def __init__(self, name, level=logging.NOTSET, time=None, count=None):
        super().__init__(name, level)

        # state storage
        self._time_limit = time
        self._count_limit = count
        stats_default = {'n_printed': 0, 'n_ignored': 0, 'last_print': 0}
        self._stats = {
            'STDOUT':  stats_default.copy(),
            'INFO':    stats_default.copy(),
            'WARNING': stats_default.copy(),
            'ERROR':   stats_default.copy(),
        }
        self._lock = threading.Lock()

        # this will determine whether the logger is already initialized
        if len(self.handlers) == 0:

            # define the default handler
            self.formatter = logging.Formatter(
                fmt=r'%(asctime)s %(name)s [%(levelname)-s]: %(message)s',
                datefmt=r'%Y-%m-%d %H:%M:%S',
            )
            handler = logging.StreamHandler(stream=sys.stderr)
            handler.setFormatter(self.formatter)
            handler.setLevel(level)
            self.addHandler(handler)
            self.addFilter(self.loggable)

            # define stdout logger
            self.to_stdout = logging.getLogger(name=f'{name}_STDOUT')
            handler_stdout = logging.StreamHandler(stream=sys.stdout)
            formatter_stdout = logging.Formatter(
                fmt=f'%(asctime)s {name}: %(message)s',
                datefmt=r'%Y-%m-%d %H:%M:%S',
            )
            handler_stdout.setFormatter(formatter_stdout)
            handler_stdout.setLevel(logging.INFO)
            self.to_stdout.addHandler(handler_stdout)
            self.to_stdout.setLevel(logging.INFO)
            self.to_stdout.addFilter(self.loggable)

        # this makes sure the levels can be changed even after the logger is initiated
        self.setLevel(level)

    def loggable(self, record):
        """Returns True if the log should be displayed. For customization, the record can be modified in place."""
        self._lock.acquire()
        loggable_time = True
        loggable_count = True
        if self.name == record.name: # its the main logger
            name = record.levelname # or logging.getLevelName(10)
        else:  # its the sub-logger
            name = re.sub(f'^{self.name}_', '', record.name)
        stats = self._stats[name]

        # check count limit
        if self._count_limit:
            if (stats['n_printed'] + stats['n_ignored']) % self._count_limit != 0:
                loggable_count = False

        # check time limit
        if self._time_limit:
            if time.time() < stats['last_print'] + self._time_limit:
                loggable_time = False
        
        is_printable = loggable_count and loggable_time
        if is_printable:
            stats['n_printed'] += 1
            stats['last_print'] = time.time()
        else:
            stats['n_ignored'] += 1
        
        self._lock.release()
        return is_printable
    
    def stdout(self, record):
        return self.to_stdout.info(record)

    def __repr__(self):
        return json.dumps(
            self.stats, 
            sort_keys=False, 
            indent=2, 
            default=str,
        ) 

    @property
    def stats(self):
        return self._stats

if __name__ == '__main__':
    # an example of logger instance
    lgr = setup_logger(name='test_logger', level=logging.DEBUG)
    lgr.debug('Parsed production page.')
    lgr.info('utility started.')
    lgr.warning('We are seeing warnings!')
    lgr.critical('This is a critical event!')
    lgr.error('Failed to send the package, error!')

    # an example of restricted logger instance
    rlgr = RestrictedLogger(name='my_logger', level=logging.WARNING, count=10)
    for i in range(30):
        rlgr.stdout(f'stdout at {i}')
        rlgr.info(f'info at {i}') # stays hidden from stats as the level=WARNING
        rlgr.error(f'error at {i}')
        time.sleep(0.01)
    print(rlgr)

    ## output:
    # 2024-11-20 09:26:26 my_logger: stdout at 0
    # 2024-11-20 09:26:26 my_logger [ERROR]: error at 0
    # 2024-11-20 09:26:26 my_logger: stdout at 10
    # 2024-11-20 09:26:26 my_logger [ERROR]: error at 10
    # 2024-11-20 09:26:26 my_logger: stdout at 20
    # 2024-11-20 09:26:26 my_logger [ERROR]: error at 20
    # {
    #   "STDOUT": {
    #     "n_printed": 0,
    #     "n_ignored": 0,
    #     "last_print": 0
    #   },
    #   "INFO": {
    #     "n_printed": 3,
    #     "n_ignored": 27,
    #     "last_print": 1732091186.505549
    #   },
    #   "WARNING": {
    #     "n_printed": 0,
    #     "n_ignored": 0,
    #     "last_print": 0
    #   },
    #   "ERROR": {
    #     "n_printed": 3,
    #     "n_ignored": 27,
    #     "last_print": 1732091186.50576
    #   }
    # }
