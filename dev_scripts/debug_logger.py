
import time
import logging

from my_utilities.convenience import setup_logger, RestrictedLogger

lgr = setup_logger(name='test_logger', level=logging.DEBUG)
lgr.debug('Parsed production page.')
lgr.info('utility started.')
lgr.warning('We are seeing warnings!')
lgr.critical('This is a critical event!')
lgr.error('Failed to send the package, error!')

# an example of restricted logger instance
rlgr = RestrictedLogger(name='test_logger', level=logging.WARNING, count=10)
for i in range(30):
    rlgr.stdout(f'stdout at {i}') # logs in `stdout`, also restricted by time/count
    rlgr.info(f'info at {i}') # stays hidden (and not counted as `ignored`) from stats as the level=WARNING
    rlgr.error(f'error at {i}')
    time.sleep(0.01)
print(rlgr)

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