
import time
import logging

from my_utilities.helpers import setup_logger, RestrictedLogger

logger = setup_logger(name='test_logger', level=logging.DEBUG)
logger.debug('Parsed production page.')
logger.info('utility started.')
logger.warning('We are seeing warnings!')
logger.critical('This is a critical event!')
logger.error('Failed to send the package, error!')

# an example of restricted logger instance
rlogger = RestrictedLogger(name='test_logger', level=logging.WARNING, count=10)
for i in range(30):
    rlogger.stdout(f'stdout at {i}') # logs in `stdout`, also restricted by time/count
    rlogger.info(f'info at {i}') # stays hidden (and not counted as `ignored`) from stats as the level=WARNING
    rlogger.error(f'error at {i}')
    time.sleep(0.01)
print(rlogger)

## output:
# [2025-10-02 11:53:27] test_logger: stdout at 0
# [2025-10-02 11:53:27] [ERROR] test_logger: error at 0
# [2025-10-02 11:53:27] test_logger: stdout at 10
# [2025-10-02 11:53:27] [ERROR] test_logger: error at 10
# [2025-10-02 11:53:27] test_logger: stdout at 20
# [2025-10-02 11:53:27] [ERROR] test_logger: error at 20
# {
#   "STDOUT": {
#     "n_printed": 3,
#     "n_ignored": 27,
#     "last_print": 1759398807.269546
#   },
#   "INFO": {
#     "n_printed": 0,
#     "n_ignored": 0,
#     "last_print": 0
#   },
#   "WARNING": {
#     "n_printed": 0,
#     "n_ignored": 0,
#     "last_print": 0
#   },
#   "ERROR": {
#     "n_printed": 3,
#     "n_ignored": 27,
#     "last_print": 1759398807.2697852
#   }
# }