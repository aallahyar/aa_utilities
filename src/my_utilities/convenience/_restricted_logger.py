import sys
import logging
import threading
import time
import json

class Logger(logging.Logger):

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
            self.to_stdout = logging.getLogger(name=f'{name}_stdout')
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
        name = record.levelname # or logging.getLevelName(10)
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
    lg = Logger(name='my_logger', level=logging.WARNING, count=10)
    for i in range(30):
        lg.stdout(f'stdout at {i}')
        lg.info(f'info at {i}') # stays hidden from stats as the level=WARNING
        lg.error(f'error at {i}')
        time.sleep(0.01)
    print(lg)

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