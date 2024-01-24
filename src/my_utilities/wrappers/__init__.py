
from importlib.util import find_spec as _find_spec
from .._logging import get_logger as _get_logger


from ._entimice import EntimICE


if _find_spec('rpy2'):
    from ._rspace import RSpace
else:
    _logger = _get_logger()
    _logger.warning('RSpace can not be loaded. RSpace requires `rpy2` package. You may install it with `pip3 install rpy2`')



