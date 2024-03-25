import torch
import einops
import datasets
import numpy as np
import pandas as pd

from opensr_degradation.naipd.main import NAIPd
from opensr_degradation.datamodel import Sensor
from opensr_degradation.utils import hq_histogram_matching
from opensr_degradation.naipd.blur import BLUR_MODEL

from typing import Any, Dict, Optional
from tqdm import tqdm

def get_s2like(
    image: torch.Tensor,
    table: dict,
    model: Optional[str] = "vae_histogram_matching"
) -> torch.Tensor:
    hr_hat = predict_table(image, table, model)

    # blur the hr_hat
    hr_hat_blur = []
    for i in range(hr_hat.shape[0]):
        blurred = BLUR_MODEL[i](hr_hat[i][None])
        hr_hat_blur.append(
            torch.nn.functional.interpolate(
                blurred[None], scale_factor=1/4, mode="bilinear", antialias=False
            ).squeeze()
        )
    hr_hat_blur = torch.stack(hr_hat_blur)
    
    return hr_hat_blur, hr_hat


def predict_table(
    image:torch.Tensor,
    table:dict,
    model: Optional[str] = "vae_histogram_matching"
) -> torch.Tensor:
    """Predict the image given a table of histograms

    Args:
        image (torch.Tensor): The image to be predicted.
        table (dict): The table of histograms.
        model (Optional[str], optional): The model to be used. Defaults 
            to "vae_histogram_matching".

    Returns:
        torch.Tensor: The predicted image.
    """
    
    image = image.clamp(0, 1)
    image.max()
    # select "model" columns
    channel, height, width = image.shape
    
    # from dict to dataframe
    table = pd.DataFrame(table)
    histograms = table.filter(regex=model).to_numpy()[0:-1].T

    # bug add negative values
    max_pixels = height * width
    missing_values = 1210000 - histograms.sum(axis=1)
    fix_hist = np.zeros_like(histograms)
    fix_hist[:, 0] = missing_values[:, None].T
    fixed_histograms = histograms + fix_hist
    
    index_name = table["bins"].to_numpy()

    # get the center of the bins
    center_bins = (index_name[1:] + index_name[0:-1]) / 2

    # repeat the center_bins given the histogram size
    corrected_image = []
    for i in range(4):
        center_bins_repeated = np.array([])
        for index, bin in enumerate(fixed_histograms[i]):        
            center_bins_repeated = np.append(
                center_bins_repeated,
                np.repeat(center_bins[index], bin)
            )            
        corrected_image.append(center_bins_repeated[None])
    corrected_image = np.stack(corrected_image)
    
    image_hat = hq_histogram_matching(
        image,
        torch.from_numpy(corrected_image)
    )

    return image_hat

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
        return self.sensor.blur_model(image, self.params)

    def harmonization(self, image: torch.Tensor) -> torch.Tensor:
        # precentiles exist in the params?
        if "percentiles" in self.params:
            if all([0 < p < 1 for p in self.params["percentiles"]]):
                raise ValueError("Percentiles must be between 0 and 100")
        return self.sensor.reflectance_model(image, self.params)
    
    def noise(self, image: torch.Tensor) -> torch.Tensor:
        return self.sensor.noise_model(image, self.params)
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hr_image = self.harmonization(image)
        
        lr_image = []
        for i in range(hr_image.shape[0]):
            lr_image.append(self.blur(hr_image[i]))
        lr_image = torch.stack(lr_image)
        if self.add_noise:
            return lr_image + self.noise(lr_image), hr_image
        return lr_image, hr_image
    
    def full_forward(self, image: torch.Tensor) -> torch.Tensor:

        # Make the image multiple of 32
        pad_x1 = (484 - (image.shape[1] % 484)) // 2
        pad_x2 = (484 - (image.shape[1] % 484)) - pad_x1
        pad_y1 = (484 - (image.shape[2] % 484)) // 2
        pad_y2 = (484 - (image.shape[2] % 484)) - pad_y1
        image_padded  = torch.nn.functional.pad(image, (pad_x1, pad_x2, pad_y1, pad_y2), mode="reflect")
        
        # Convert to the image to the right shape
        n, m = image_padded.shape[1]//484, image_padded.shape[2]//484
        image_padded_r  = einops.rearrange(
            image_padded,
            "c (h1 h2) (w1 w2) -> (h1 w1) c h2 w2", 
            h1=n, w1=m
        )

        # Apply the degradation model
        lr_container = []
        hr_container = []
        for i in range(image_padded_r.shape[0]):
            lr, hr = self.forward(image_padded_r[i])
            lr_container.append(lr)
            hr_container.append(hr)
        lr_image_padded_r_r = torch.stack(lr_container)
        hr_image_padded_r_r = torch.stack(hr_container)        

        # Go back to the LR original shape
        lr_image_padded_r_grid  = einops.rearrange(
            lr_image_padded_r_r,
            "(p1 p2) s c h w -> s c (p1 h) (p2 w)",
            p1=n, p2=m
        )
        mm = lr_image_padded_r_grid.shape[-1]    
        lr_image_padded_r_grid_npad = lr_image_padded_r_grid[
            :, :, (pad_x1//4):(mm-pad_x2//4), (pad_y1//4):(mm-pad_y2//4)
        ]
        
        # Go back to the HR original shape
        hr_image_padded_r_grid  = einops.rearrange(
            hr_image_padded_r_r,
            "(p1 p2) s c h w -> s c (p1 h) (p2 w)",
            p1=n, p2=m
        )
        mm = hr_image_padded_r_grid.shape[-1]
        hr_image_padded_r_grid_npad = hr_image_padded_r_grid[
            :, :, pad_x1:(mm-pad_x2), pad_y1:(mm-pad_y2)
        ]
        
        return lr_image_padded_r_grid_npad, hr_image_padded_r_grid_npad

    def full_table(self, hr: torch.Tensor) -> pd.DataFrame:        
        bins = np.linspace(0, 1., 1001)
        reflectance_names = [
            "identity",
            "gamma_lognormal_10",
            "gamma_lognormal_25",
            "gamma_lognormal_50",
            "gamma_lognormal_75",
            "gamma_lognormal_90",
            "gamma_multivariate_normal_10",
            "gamma_multivariate_normal_25",
            "gamma_multivariate_normal_50",
            "gamma_multivariate_normal_75",
            "gamma_multivariate_normal_90",
            "unet_histogram_matching",
            "vae_histogram_matching"
        ]
        bands = ["red", "green", "blue", "nir"]
        dataset = pd.DataFrame({"bins": bins})
        s, c, h, w = hr.shape
        for i in range(s):
            for j in range(c):
                hr_np, _ = np.histogram(hr[i, j].flatten(), bins=bins)
                dataset[f"lr__{bands[j]}__{reflectance_names[i]}"] = np.append(hr_np, -999)
        return dataset    

