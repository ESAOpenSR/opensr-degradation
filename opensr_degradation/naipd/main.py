from typing import Any, List, Optional, Union
import pkg_resources

import torchvision
import torch

from opensr_degradation.datamodel import Sensor
from opensr_degradation.naipd.blur import blur
from opensr_degradation.naipd.harmonization import harmonization, SimpleVAE
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


def naip_vae_model(device: Union[str, torch.device] = "cpu") -> SimpleVAE:
    file = pkg_resources.resource_filename(
        "opensr_degradation", "naipd/models/model_vae.pt"
    )
    reflectance_model = SimpleVAE()
    reflectance_model.load_state_dict(torch.load(file))
    reflectance_model.eval()
    reflectance_model.to(device)

    return reflectance_model


def naip_unet_model(device: Union[str, torch.device] = "cpu") -> torch.jit.ScriptModule:
    file = pkg_resources.resource_filename(
        "opensr_degradation", "naipd/models/model_unet.pt"
    )
    reflectance_model = torch.jit.load(file, map_location=device)
    reflectance_model.eval()
    
    return reflectance_model
                                       