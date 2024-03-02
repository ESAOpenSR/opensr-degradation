"""A set of method to make NAIP look like Sentinel-2"""

from importlib import metadata
from opensr_degradation.main import pipe
from opensr_degradation.naipd.main import NAIPd, naip_vae_model, naip_unet_model

def _get_version() -> str:
    try:
        return metadata.version("opensr-degradation")
    except ModuleNotFoundError:  # pragma: no cover
        return "unknown"


__version__ = _get_version()
