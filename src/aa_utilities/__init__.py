# getting the current version of the package
import importlib.metadata as _metadata
__version__ = _metadata.version('aa_utilities')

# expose package-wide configuration (repository URL, logging level, etc.)
from ._configurations import configs

# lazily expose submodules (e.g. `import aa_utilities; aa_utilities.graphics.heatmap(...)`)
# without eagerly importing their (sometimes heavy/optional) dependencies at package import time.
_SUBMODULES = frozenset({'computation', 'graphics', 'helpers', 'loggers', 'stats', 'storage', 'wrappers'})


def __getattr__(name):
    if name in _SUBMODULES:
        import importlib

        module = importlib.import_module(f'.{name}', __name__)
        globals()[name] = module  # cache so subsequent access skips __getattr__
        return module
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | _SUBMODULES)
