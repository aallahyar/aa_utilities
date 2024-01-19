

from importlib import resources

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


# getting the current version
try:
    import importlib.metadata
except ModuleNotFoundError:
    import importlib_metadata
__version__ = importlib.metadata.version("my_utilities")

# Read URL of the Real Python feed from config file
_cfg = tomllib.loads(resources.read_text("my_utilities", "config.toml"))
URL = _cfg["repository"]["url"]
