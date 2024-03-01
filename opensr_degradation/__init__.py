"""A set of method to make NAIP look like Sentinel-2"""

from importlib import metadata
from opensr_degradation.main import pipe

def _get_version() -> str:
    try:
        return metadata.version("opensr-degradation")
    except ModuleNotFoundError:  # pragma: no cover
        return "unknown"


__version__ = _get_version()
