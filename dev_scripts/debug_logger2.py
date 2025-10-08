import logging
import threading
import time
import sys

# Optional color support via colorama
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
    COLOR_ENABLED = True
except ImportError:
    COLOR_ENABLED = False

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


class ColoredFormatter(logging.Formatter):
    """Colors only the levelname; falls back to plain text if colorama is missing."""
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


class ThrottleFilter(logging.Filter):
    """
    Simple throttling filter.

    - count_n: allow the 1st message immediately, then every Nth attempt thereafter
               (attempt indices 1, 1+N, 1+2N, ...).
    - time_s: require at least time_s seconds since the last printed message.
    - mode: "and" -> both count/time must allow; "or" -> either allows.

    Attach this filter to a handler to throttle that handler's output.
    Thread-safe.
    """
    def __init__(self, name: str = "", *, count_n: int | None = None, time_s: float | None = None, mode: str = "and"):
        super().__init__(name=name)
        self.count_n = count_n if (count_n is None or count_n > 0) else None
        self.time_s = time_s if (time_s is None or time_s > 0) else None
        self.mode = "and" if mode not in ("and", "or") else mode

        self._lock = threading.Lock()
        self._attempts = 0
        self._last_print = 0.0
        self._printed = 0
        self._ignored = 0

    def filter(self, record: logging.LogRecord) -> bool:
        now = time.time()
        with self._lock:
            self._attempts += 1

            allow_count = True
            if self.count_n is not None:
                # First attempt passes (attempts==1), then every Nth: (attempt-1) % N == 0
                allow_count = ((self._attempts - 1) % self.count_n) == 0

            allow_time = True
            if self.time_s is not None:
                allow_time = now >= (self._last_print + self.time_s)

            allow = (allow_count and allow_time) if self.mode == "and" else (allow_count or allow_time)

            if allow:
                self._printed += 1
                self._last_print = now
            else:
                self._ignored += 1

            return allow

    def stats(self) -> dict:
        with self._lock:
            return {
                "attempts": self._attempts,
                "printed": self._printed,
                "ignored": self._ignored,
                "last_print": self._last_print,
                "count_n": self.count_n,
                "time_s": self.time_s,
                "mode": self.mode,
            }


def make_colored_logger_with_throttle(
    name: str = "demo.throttle",
    level: int = logging.INFO,
    *,
    count_n: int | None = None,
    time_s: float | None = None,
    mode: str = "and",
) -> tuple[logging.Logger, ThrottleFilter]:
    """
    Build a logger that prints to stderr with color and throttling.
    Returns (logger, throttle_filter) so you can inspect stats.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handler if this is called multiple times
    existing = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stderr]
    if existing:
        handler = existing[0]
    else:
        handler = logging.StreamHandler(stream=sys.stderr)
        logger.addHandler(handler)

    handler.set_name("stderr_main") # retrievable via `h.get_name()` if needed
    handler.setLevel(logging.NOTSET)
    handler.setFormatter(ColoredFormatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    # Remove any prior ThrottleFilter to avoid stacking effects
    for f in list(handler.filters):
        if isinstance(f, ThrottleFilter):
            handler.removeFilter(f)

    throttle = ThrottleFilter(count_n=count_n, time_s=time_s, mode=mode)
    handler.addFilter(throttle)

    return logger, throttle


if __name__ == "__main__":
    # Example 1: Count-based only (first, then every 5th)
    log, thr = make_colored_logger_with_throttle(count_n=5, time_s=None, mode="and")
    for i in range(1, 16):
        log.info(f"count-only attempt {i}")
    print("Stats:", thr.stats())

    # Example 2: Time-based only (at most one message per 2 seconds)
    log2, thr2 = make_colored_logger_with_throttle(name="demo.time", count_n=None, time_s=2.0, mode="and")
    for i in range(5):
        log2.warning(f"time-only attempt {i}")
        time.sleep(0.5)  # Only every 4th will likely print (2.0s window)

    print("Stats (time):", thr2.stats())

    # Example 3: Combine both with mode="or" (print if either gate allows)
    log3, thr3 = make_colored_logger_with_throttle(name="demo.or", count_n=3, time_s=1.5, mode="or")
    for i in range(10):
        log3.error(f"combined OR attempt {i}")
        time.sleep(0.4)

    print("Stats (combined):", thr3.stats())