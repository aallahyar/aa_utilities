

from importlib import resources as _resources

try:
    import tomllib as _tomllib
except ModuleNotFoundError:
    import tomli as _tomllib


# getting the current version of the package
try:
    import importlib.metadata as _metadata
except ModuleNotFoundError:
    import importlib_metadata as _metadata
__version__ = _metadata.version("my_utilities")

# Read URL of the Real Python feed from config file
_cfg = _tomllib.loads(_resources.read_text("my_utilities", "config.toml"))
URL = _cfg["repository"]["url"]


