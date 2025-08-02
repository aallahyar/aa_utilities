

# getting the current version of the package
try:
    import importlib.metadata as _metadata
except ModuleNotFoundError:
    import importlib_metadata as _metadata
__version__ = _metadata.version("my_utilities")

# Read URL of the Real Python feed from config file
from ._configuration import _configurations


# making modules visible to user
# from . import (
#     graphics, 
#     wrappers, 
#     computations, 
#     convenience,
# )


