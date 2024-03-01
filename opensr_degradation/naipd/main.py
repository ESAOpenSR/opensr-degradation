from typing import Any, List, Optional

import torchvision

from opensr_degradation.datamodel import Sensor
from opensr_degradation.naipd.blur import blur
from opensr_degradation.naipd.harmonization import harmonization
from opensr_degradation.naipd.noise import noise

class NAIPd(Sensor):
    """A degraded version of NAIP imagery.

    Args:
        Sensor (Sensor): A NAIP collection.
    """

    bands: List[str] = ["R", "G", "B", "NIR"]
    sensor_name: str = "NAIP"
    resolution: List[int] = [3.0, 2.9, 2.9, 3.4]
    blur_model: Optional[Any] = blur
    reflectance_model: Optional[Any] = harmonization
    noise_model: Optional[Any] = noise
    center_wavelength: Optional[List[float]] = None
    full_width_half_max: Optional[List[float]] = None
