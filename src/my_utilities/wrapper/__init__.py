
from importlib.util import find_spec as _find_spec
from ..convenience import setup_logger


from ._entimice import EntimICE


if _find_spec('rpy2'):
    from ._rspace import RSpace
else:
    _logger = setup_logger(name=__name__)
    _logger.warning('RSpace can not be loaded. RSpace requires `rpy2` package. You may install it with `pip3 install rpy2`')



