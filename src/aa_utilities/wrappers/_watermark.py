import inspect
import platform
import types
from datetime import datetime
from multiprocessing import cpu_count
from zoneinfo import ZoneInfo

import IPython


def watermark(
        author='Unknown Author',
        email='Unknown Email',
        timezone='Europe/Stockholm',
        namespace=None,
    ):
    """Generates and returns a watermark string with system and environment info.
    The implementation is inspired by the `watermark` package:
    https://github.com/rasbt/watermark

    Parameters
    ----------
    author : str
        Author name.
    email : str
        Author email.
    timezone : str or None
        IANA timezone string for the local time stamp (e.g. 'Europe/Stockholm').
        Pass None to omit the time entirely.
    namespace : dict or None
        Pass globals() to include versions of all imported packages.

    Example:
        ```
        from aa_utilities.wrappers import watermark as aa_watermark
        aa_watermark(
            author="Amin Allahyar",
            email='Amin.Allahyar@astrazeneca.com',
            timezone='Europe/Stockholm',
            namespace=globals(),
        )
        ```
    """
    sections = []

    # Author info — always first
    sections.append({
        "Author": author,
        "Email": email,
    })

    # Date / time
    date_section = {"Date": datetime.now().strftime("%Y-%m-%d")}
    if timezone is not None:
        date_section["Time"] = datetime.now(tz=ZoneInfo(timezone)).strftime("%H:%M:%S %Z")
    sections.append(date_section)

    # Python and IPython versions
    sections.append(_get_pyversions())

    # System info
    sections.append(_get_sysinfo())

    # Imported package versions
    if namespace is not None:
        sections.append(_get_import_versions(namespace))

    return _format_sections(sections)


def _get_pyversions():
    return {
        "Python implementation": platform.python_implementation(),
        "Python version": platform.python_version(),
        "IPython version": IPython.__version__,
    }


def _get_sysinfo():
    return {
        "Compiler": platform.python_compiler(),
        "OS": platform.system(),
        "Release": platform.release(),
        "Machine": platform.machine(),
        "Processor": platform.processor(),
        "CPU cores": cpu_count(),
        "Architecture": platform.architecture()[0],
    }


def _get_import_versions(namespace):
    modules = {v for v in namespace.values() if isinstance(v, types.ModuleType)}
    modules.update(
        inspect.getmodule(v)
        for v in namespace.values()
        if inspect.isclass(v) or inspect.isfunction(v)
    )
    modules.discard(None)

    versions = {}
    for mod in sorted(modules, key=lambda m: m.__name__):
        pkg_name = mod.__name__.split(".")[0]
        version = getattr(mod, "__version__", None)
        if version is None:
            try:
                import importlib.metadata as _meta
                version = _meta.version(pkg_name)
            except Exception:
                continue
        versions[pkg_name] = version
    return versions


def _format_sections(sections):
    blocks = []
    for section in sections:
        if not section:
            continue
        width = max(len(k) for k in section)
        block = "\n".join(f"{k.ljust(width)}: {v}" for k, v in section.items())
        blocks.append(block)
    return "\n\n".join(blocks)
