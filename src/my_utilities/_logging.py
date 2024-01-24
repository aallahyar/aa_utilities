
# inspired from pairtools: https://github.com/open2c/pairtools/blob/master/pairtools/_logging.py

import logging as _logging

from ._configuration import _configurations

_loggers = {}

def get_logger(name="my_utilities", level=None):
    global _loggers

    if name not in _loggers:
        _loggers[name] = _logging.getLogger(name)
        _loggers[name].propagate = False

        handler = _logging.StreamHandler()
        formatter = _logging.Formatter(
            '%(asctime)s %(name)s [%(levelname)s]: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        handler.setFormatter(formatter)
        _loggers[name].addHandler(handler)
        
        if level is None:
            level=_logging.DEBUG if _configurations['debug'] else _logging.INFO
        _loggers[name].setLevel(level)
    
    return _loggers[name]
