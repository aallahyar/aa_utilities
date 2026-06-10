# getting the current version of the package
import importlib.metadata as _metadata
__version__ = _metadata.version('aa_utilities')

# Read URL of the Real Python feed from config file
from ._configurations import configs


# making modules visible to user
# from . import (
#     graphics,
#     wrappers,
#     computations,
#     convenience,
# )
