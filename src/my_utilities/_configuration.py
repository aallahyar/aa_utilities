
from importlib import resources as _resources

try:
    import tomllib as _tomllib
except ModuleNotFoundError:
    import tomli as _tomllib

_configurations = _tomllib.loads(_resources.read_text("my_utilities", "configurations.toml"))