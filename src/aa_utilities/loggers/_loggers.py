import copy
import sys
import logging
import os
import re
import threading
import time
import json

# from datetime import datetime
# from logging.handlers import TimedRotatingFileHandler

from .._configurations import configs

# marks the handler installed by `setup_logger`, so repeated calls update it instead of stacking another
_HANDLER_TAG = '_aa_utilities_handler'

# Log format
LOG_FORMAT = r'[%(asctime)s] [%(levelname)s] %(name)s: %(message)s'
DATE_FORMAT = r'%Y-%m-%d %H:%M:%S'


def _color_supported(stream) -> bool:
    """Whether ANSI colors should be used for `stream`: honors NO_COLOR, and allows terminals and Jupyter kernels."""
    if os.environ.get('NO_COLOR'):
        return False
    if getattr(stream, 'isatty', lambda: False)():
        return True
    return 'ipykernel' in sys.modules  # Jupyter streams are not ttys, but render ANSI


class ColoredFormatter(logging.Formatter):
    """Colors the level name, and also the message for WARNING and above, using plain ANSI escapes."""

    RESET = '\033[0m'
    COLORS = {
        logging.DEBUG: '\033[90m',
        logging.INFO: '\033[32m',
        logging.WARNING: '\033[33m',
        logging.ERROR: '\033[31m',
        logging.CRITICAL: '\033[35m',
    }

    def __init__(self, fmt=None, datefmt=None, color=True):
        super().__init__(fmt, datefmt)
        self.color = color

    def format(self, record):
        if not self.color:
            return super().format(record)

        # format a copy, so other handlers receiving the same record are unaffected
        record = copy.copy(record)
        color = self.COLORS.get(record.levelno, '')
        record.levelname = f'{color}{record.levelname}{self.RESET}'
        if record.levelno >= logging.WARNING:
            record.msg = f'{color}{record.getMessage()}{self.RESET}'
            record.args = None
        return super().format(record)


# Example of a logger that stores in a file
# file_handler = logging.FileHandler("app.log", mode="a", encoding="utf-8")
# file_handler.setLevel(logging.INFO)

# Example of file handler with rotation
# handler = TimedRotatingFileHandler(
#     filename=log_path,
#     when="W0",   # time-based rotation trigger. roll over at the specified weekday, 0=Monday
#     interval=1,        # rotate every 1 'when' unit. e.g. when="H", interval=6 rotates every 6 hours.
#     backupCount=7,     # keep 7 old log files. Older files beyond this count are deleted.
#     encoding="utf-8",
#     utc=False          # use local time for rotation
# )
# handler = RotatingFileHandler(
#     filename=log_path,
#     mode='a', # adds new logs to the end and preserves existing content.
#     # maxBytes=2_000_000, # Size threshold that triggers rotation. Max 2MB per file.
#     # backupCount=3, # If maxBytes is zero, rollover never occurs.
#     # Rollover occurs whenever the current log file is nearly maxBytes in length.
#     # If backupCount is >= 1, the system will successively create new files with the same pathname
#     # as the base file, but with extensions ".1", ".2" etc. Existing backups are shifted up in the sequence
#     delay=True, # If True, the file is not opened until the first log message is emitted.
#     encoding='utf-8',
# )


def setup_logger(name='Unknown', level=None, color=None, propagate=False) -> logging.Logger:
    """Returns the standard `logging.Logger` called `name`, with a (colored) stderr handler attached.

    Meant for applications (scripts, notebooks); library modules should use `logging.getLogger(__name__)`.
    Calling it again on the same name updates the existing handler instead of adding another.

    Args:
        name: Logger name.
        level: Logging level (e.g., 'DEBUG', logging.INFO). Defaults to the logger's current level,
            or `configs.log.level` if it has none yet.
        color: Force colors on/off. Defaults to auto: on for terminals and Jupyter, off if `NO_COLOR` is set.
        propagate: Whether records are also passed to the handlers of ancestor loggers (e.g., root).
            Off by default to avoid duplicated lines when the root logger has its own handler.

    Example:
        log = setup_logger('my_analysis', level='DEBUG')
        log.warning('shown in yellow')
    """
    logger = logging.getLogger(name)
    if level is None:
        level = logger.level or configs.log.level
    logger.setLevel(level)
    logger.propagate = propagate

    handler = next((h for h in logger.handlers if getattr(h, _HANDLER_TAG, False)), None)
    if handler is None:
        handler = logging.StreamHandler(stream=sys.stderr)
        setattr(handler, _HANDLER_TAG, True)
        logger.addHandler(handler)
    if color is None:
        color = _color_supported(handler.stream)
    handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT, color=color))  # handler level stays NOTSET: the logger's level decides

    return logger


class RestrictedLogger(logging.Logger):
    def __init__(self, name, level=configs.log.level, time=None, count=None):
        super().__init__(name, level)

        # state storage
        self._time_limit = time
        self._count_limit = count
        stats_default = {'n_printed': 0, 'n_ignored': 0, 'last_print': 0}
        self._stats = {
            'STDOUT': stats_default.copy(),
            'DEBUG': stats_default.copy(),
            'INFO': stats_default.copy(),
            'WARNING': stats_default.copy(),
            'ERROR': stats_default.copy(),
            'CRITICAL': stats_default.copy(),
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
            handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT, color=_color_supported(handler.stream)))
            handler.setLevel(level)
            self.addHandler(handler)
            self.addFilter(self.loggable)

            # define stdout logger: it prints to stdout regardless of the level (with an added timestamp)
            self.to_stdout = logging.getLogger(name=f'{name}_STDOUT')
            handler_stdout = logging.StreamHandler(stream=sys.stdout)
            handler_stdout.setFormatter(
                ColoredFormatter(
                    f'[%(asctime)s] {name}: %(message)s',
                    datefmt=DATE_FORMAT,
                    color=_color_supported(handler_stdout.stream),
                )
            )
            handler_stdout.setLevel(logging.INFO)
            self.to_stdout.addHandler(handler_stdout)
            self.to_stdout.setLevel(logging.INFO)
            self.to_stdout.addFilter(self.loggable)

        # this makes sure the levels can be changed even after the logger is initiated
        self.setLevel(level)

    def loggable(self, record):
        """Returns True if the log should be displayed. For customization, the record can be modified in place."""
        with self._lock:  # ensures the lock is released even if an exception occurs below
            loggable_time = True
            loggable_count = True
            if self.name == record.name:  # its the main logger
                name = record.levelname  # or logging.getLevelName(10)
            else:  # its the sub-logger
                name = re.sub(f'^{self.name}_', '', record.name)
            stats = self._stats[name]

            # check count limit
            if self._count_limit is not None:
                total_attempted = stats['n_printed'] + stats['n_ignored']
                if total_attempted % self._count_limit != 0:
                    loggable_count = False

            # check time limit
            if self._time_limit is not None:
                if time.time() < stats['last_print'] + self._time_limit:
                    loggable_time = False

            is_printable = loggable_count and loggable_time
            if is_printable:
                stats['n_printed'] += 1
                stats['last_print'] = time.time()
            else:
                stats['n_ignored'] += 1

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
