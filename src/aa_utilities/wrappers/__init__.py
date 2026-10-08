import logging as _logging
from importlib.util import find_spec as _find_spec


if _find_spec('rpy2'):
    from ._rspace import RSpace, RWarning
else:
    _logging.getLogger(__name__).warning(
        'RSpace can not be loaded. RSpace requires `rpy2` package. You may install it with `pip3 install rpy2`'
    )


