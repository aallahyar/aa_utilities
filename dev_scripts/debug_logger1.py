import logging
import threading
import time
import sys
from typing import Optional, Literal

# Optional color support
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

# Common formats
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Formatter that colors only the levelname; leaves other fields intact."""
    if COLOR_ENABLED:
        COLORS = {
            logging.DEBUG:   Fore.LIGHTBLACK_EX,
            logging.INFO:    Fore.GREEN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR:   Fore.RED,
            logging.CRITICAL: Fore.MAGENTA,
        }

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        try:
            if COLOR_ENABLED:
                color = self.COLORS.get(record.levelno, "")
                record.levelname = f"{color}{original}{Style.RESET_ALL}"
            return super().format(record)
        finally:
            record.levelname = original


class ThrottlePolicy:
    """
    Encapsulates count/time throttling with AND/OR composition.

    - count_n: print first attempt immediately, then every Nth attempt thereafter
               (attempts index: 1, 1+N, 1+2N, ...).
    - time_s: minimum seconds between prints (sliding window).
    - mode: "and" -> both gates must pass; "or" -> either gate can pass.
    """
    def __init__(
        self,
        *,
        count_n: Optional[int] = None,
        time_s: Optional[float] = None,
        mode: Literal["and", "or"] = "and",
        time_fn=time.time,
    ):
        self.count_n = count_n if (count_n is None or count_n > 0) else None
        self.time_s = time_s if (time_s is None or time_s > 0) else None
        self.mode = mode
        self.time_fn = time_fn

        self._lock = threading.Lock()
        self._attempts = 0
        self._last_print = 0.0

    def allow(self) -> bool:
        with self._lock:
            now = self.time_fn()
            self._attempts += 1

            allow_count = True
            if self.count_n is not None:
                allow_count = ((self._attempts - 1) % self.count_n) == 0

            allow_time = True
            if self.time_s is not None:
                allow_time = now >= (self._last_print + self.time_s)

            allow = (allow_count and allow_time) if self.mode == "and" else (allow_count or allow_time)
            if allow:
                self._last_print = now
            return allow


class ThrottledLogger:
    """
    A logger-like wrapper that applies throttling before forwarding to an underlying logger.

    - Does not modify handlers or formatters.
    - Safe for multi-threaded use.
    - Preserves logging API: debug/info/warning/error/exception/critical/log
    """

    def __init__(
        self,
        base_logger: logging.Logger,
        *,
        count_n: Optional[int] = None,
        time_s: Optional[float] = None,
        mode: Literal["and", "or"] = "and",
    ):
        self._logger = base_logger
        self._policy = ThrottlePolicy(count_n=count_n, time_s=time_s, mode=mode)
        self._printed = 0
        self._ignored = 0
        self._stats_lock = threading.Lock()

    # Core
    def log(self, level: int, msg, *args, **kwargs):
        if self._policy.allow():
            with self._stats_lock:
                self._printed += 1
            return self._logger.log(level, msg, *args, **kwargs)
        else:
            with self._stats_lock:
                self._ignored += 1
            return

    # Convenience methods matching logging.Logger API
    def debug(self, msg, *args, **kwargs):
        return self.log(logging.DEBUG, msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self.log(logging.INFO, msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self.log(logging.WARNING, msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self.log(logging.ERROR, msg, *args, **kwargs)

    def exception(self, msg, *args, exc_info=True, **kwargs):
        # Ensure exceptions include traceback by default
        return self.log(logging.ERROR, msg, *args, exc_info=exc_info, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self.log(logging.CRITICAL, msg, *args, **kwargs)

    # Useful extras
    @property
    def name(self) -> str:
        return self._logger.name

    def setLevel(self, level: int):
        self._logger.setLevel(level)

    def get_stats(self):
        with self._stats_lock:
            return {"n_printed": self._printed, "n_ignored": self._ignored}

    def __repr__(self):
        return f"<ThrottledLogger name={self.name} stats={self.get_stats()}>"

    # Access to underlying logger if needed
    @property
    def underlying(self) -> logging.Logger:
        return self._logger


class ColorizedLogger:
    """
    Decorator that applies a ColoredFormatter to all existing StreamHandlers of a logger.
    It does not change throttling, levels, or propagation. It is idempotent.

    - If a handler already has a ColoredFormatter, it is left as-is.
    - Non-stream handlers (e.g., file handlers) are also colorized if desired.
    """

    def __init__(self, base_logger: logging.Logger, *, color_enabled: bool = COLOR_ENABLED):
        self._logger = base_logger
        self._color_enabled = color_enabled
        if color_enabled:
            self._apply_color_formatters()

    def _apply_color_formatters(self):
        for h in self._logger.handlers:
            # Skip if formatter already colorizes levelname
            if isinstance(h.formatter, ColoredFormatter):
                continue

            # Keep the same format strings if present; else fallback
            fmt = LOG_FORMAT
            datefmt = DATE_FORMAT
            if h.formatter is not None:
                # Try to preserve the formatter's format string
                try:
                    fmt = h.formatter._fmt  # using internal attribute for simplicity
                except Exception:
                    pass
                try:
                    datefmt = h.formatter.datefmt or DATE_FORMAT
                except Exception:
                    pass

            h.setFormatter(ColoredFormatter(fmt, datefmt=datefmt))

    # Proxy logging methods to the underlying logger
    def log(self, level: int, msg, *args, **kwargs):
        return self._logger.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        return self._logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        return self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        return self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        return self._logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):
        return self._logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        return self._logger.critical(msg, *args, **kwargs)

    @property
    def name(self) -> str:
        return self._logger.name

    def setLevel(self, level: int):
        self._logger.setLevel(level)

    @property
    def underlying(self) -> logging.Logger:
        return self._logger

    def __repr__(self):
        return f"<ColorizedLogger name={self.name} color={self._color_enabled}>"


def make_base_logger(
    name: str,
    level: int = logging.INFO,
    to_stderr: bool = True,
    to_stdout: bool = False,
    fmt: str = LOG_FORMAT,
    datefmt: str = DATE_FORMAT,
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if called multiple times
    streams = {getattr(h, "stream", None) for h in logger.handlers if isinstance(h, logging.StreamHandler)}
    if to_stderr and sys.stderr not in streams:
        h = logging.StreamHandler(sys.stderr)
        h.setLevel(logging.NOTSET)
        h.setFormatter(logging.Formatter(fmt, datefmt=datefmt))
        logger.addHandler(h)

    if to_stdout and sys.stdout not in streams:
        h = logging.StreamHandler(sys.stdout)
        h.setLevel(logging.NOTSET)
        # Simpler stdout line, no level by default; can keep fmt if preferred
        h.setFormatter(logging.Formatter(f"[%(asctime)s] {name}: %(message)s", datefmt=datefmt))
        logger.addHandler(h)

    return logger


# Example composition:
# 1) Build a base logger with stderr handler
base = make_base_logger("my.module", level=logging.INFO, to_stderr=True, to_stdout=False)

# # 2) Wrap with throttling (e.g., every 10th or at least 2 seconds between prints, AND-composed)
throttled = ThrottledLogger(base, count_n=10, time_s=2.0, mode="or")

# # 3) Colorize the throttled logger (keeps throttling intact)
# colored_throttled = ColorizedLogger(throttled.underlying)  # or ColorizedLogger(base)

# # Use whichever interface you prefer:
# throttled.info("This respects throttling and the underlying handlers.")
# colored_throttled.info("This is colored; throttling depends on where you place the wrapper.")

# # If you want both throttling and color while using the same reference:
# # Wrap in color first, then throttle the colored logger (functionally equivalent)
# colored_then_throttled = ThrottledLogger(colored_throttled.underlying, count_n=10, time_s=2.0, mode="and")
# colored_then_throttled.info("Throttled, and the handlers are colorized.")

# base.debug('Parsed production page.')
# base.info('utility started.')
# base.warning('We are seeing warnings!')
# base.critical('This is a critical event!')
# base.error('Failed to send the package, error!')

# an example of restricted logger instance
# rlogger = RestrictedLogger(name='test_logger', level=logging.WARNING, count=10)
for i in range(30):
    # throttled.stdout(f'stdout at {i}') # logs in `stdout`, also restricted by time/count
    throttled.info(f'info at {i}') # stays hidden (and not counted as `ignored`) from stats as the level=WARNING
    throttled.error(f'error at {i}')
    time.sleep(0.01)
print(throttled)