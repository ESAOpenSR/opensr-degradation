import torch

from opensr_degradation.naipd.main import NAIPd
from opensr_degradation.datamodel import Sensor
from opensr_degradation.utils import hq_histogram_matching

from typing import Any, Dict, Optional

class pipe:
    def __init__(
        self,
        sensor: Sensor,
        add_noise: Optional[bool] = True,
        params: Optional[Dict[str, Any]] = {},
    ):
        if sensor == "naip_d":
            self.sensor = NAIPd()
        else:
            raise ValueError("Model not found")
        
        self.add_noise = add_noise
        self.params = params

    def blur(self, image: torch.Tensor) -> torch.Tensor:
        return self.sensor.blur_model(image, **self.params)

    def harmonization(self, image: torch.Tensor) -> torch.Tensor:
        # precentiles exist in the params?
        if "percentiles" in self.params:
            if all([0 < p < 1 for p in self.params["percentiles"]]):
                raise ValueError("Percentiles must be between 0 and 100")
        return self.sensor.reflectance_model(image, self.params)
    
    def noise(self, image: torch.Tensor) -> torch.Tensor:
        return self.sensor.noise_model(image, **self.params)
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        lr_image = self.blur(image)
        lr_image = self.harmonization(lr_image)        
        
        # repeat the harmonization to match the LR image
        hr_image = torch.repeat_interleave(
            image[None],
            lr_image.shape[0],
            dim=0
        )
        
        hr_image = hq_histogram_matching(hr_image, lr_image)
        if self.add_noise:
            return lr_image + self.noise(lr_image), hr_image
        return lr_image, hr_image