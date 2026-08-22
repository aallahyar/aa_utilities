# getting the current version of the package
import importlib.metadata as _metadata
__version__ = _metadata.version('aa_utilities')

# expose package-wide configuration (repository URL, logging level, etc.)
from ._configurations import configs


# making modules visible to user
# from . import (
#     graphics,
#     wrappers,
#     computations,
#     convenience,
# )
