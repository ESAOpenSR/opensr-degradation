from typing import Optional, Union, List

import pathlib

import numpy as np
import pkg_resources
import scipy.stats as stats
import skimage.exposure
import torch
import torchvision
from torch import nn

from opensr_degradation.utils import hq_histogram_matching
from collections import OrderedDict

class SimpleVAE(torch.nn.Module):
    def __init__(self, input_shape=(4, 120), latent_dim=20):
        super().__init__()
        self.input_dim = torch.prod(torch.tensor(input_shape))
        self.latent_dim = latent_dim

        # Replace linear layers with convolutions
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.input_dim),
            nn.Unflatten(1, input_shape),
            nn.Sigmoid(),
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


def harmonization_01(
    image: torch.Tensor,
    percentiles: Optional[float] = 50,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs
) -> torch.Tensor:
    
    # Set the random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Convert the image to 8-bit
    image = (image * 255).to(torch.uint8)

    # Set the log-normal model
    params = (0.08328806939205786, -0.5100939882738114, 0.890445913241666)
    model = stats.lognorm(*params[:-2], loc=params[-2], scale=params[-1])

    # Make a sample of the model
    sample = model.rvs(size=100000)
    model_value = np.percentile(sample, percentiles)

    def power_law(x, gamma, k: int = 255):
        return ((x / k) ** (1 / gamma))

    # Apply the power law to the image
    return power_law(image, model_value).to(device)


def harmonization_02(
    image: torch.Tensor,
    percentiles: Optional[float] = 50,
    seed: Optional[int] = None,
    device: Union[str, torch.device] = "cpu",
    **kwargs
) -> torch.Tensor:

    # Convert the image to 8-bit
    image = (image * 255).to(torch.uint8)

    # Set the random seed
    if seed is not None:
        torch.manual_seed(seed)

    # Set the log-normal model
    mean = np.array([0.33212655, 0.32608668, 0.35346831, 0.45705735])
    std = np.array(
        [
            [0.0059202, 0.00507745, 0.00445213, 0.00364854],
            [0.00507745, 0.00494151, 0.00445722, 0.00381408],
            [0.00445213, 0.00445722, 0.00485922, 0.00329423],
            [0.00364854, 0.00381408, 0.00329423, 0.01427048],
        ]
    )
    model = stats.multivariate_normal(mean=mean, cov=std)

    def power_law(x, gamma, k: int = 255):
        return ((x / k) ** (1 / gamma))

    # Make a sample of the model
    sample = model.rvs(size=100000)
    model_value = np.percentile(sample, percentiles)

    # Apply the power law to the image
    return power_law(image, model_value).to(device)


def harmonization_03(
    image: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
    unet_reflectance_model: Optional[torch.jit.ScriptModule] = None,
    **kwargs
) -> torch.Tensor:

    if unet_reflectance_model is None:
        file = pkg_resources.resource_filename(
            "opensr_degradation", "naipd/models/model_unet.pt"
        )
        reflectance_model = torch.jit.load(file, map_location=device)
        reflectance_model.eval()
    else:
        reflectance_model = unet_reflectance_model

    # Pad the image to make it divisible by 32
    to_pad = 32 - (image.shape[-1] % 32)
    padded_image = torch.nn.functional.pad(image, (0, to_pad, 0, to_pad))

    # Perform the inference
    with torch.no_grad():
        tensor = padded_image.to(device).float()
        tensor_hat = reflectance_model(tensor[None]).squeeze()
        tensor_hat = tensor_hat[:, : image.shape[1], : image.shape[2]]

    return hq_histogram_matching(image, tensor_hat).to(device)


def harmonization_04(
    image: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
    vae_reflectance_model: Optional[torch.nn.Module] = None,
    **kwargs
) -> torch.Tensor:

    # Set the model
    if vae_reflectance_model is None:
        file = pkg_resources.resource_filename(
            "opensr_degradation", "naipd/models/model_vae.pt"
        )
        reflectance_model = SimpleVAE()
        reflectance_model.load_state_dict(torch.load(file))
        reflectance_model.eval()
        reflectance_model.to(device)
    else:
        reflectance_model = vae_reflectance_model

    # Set the image to the right device
    image = image.to(device)

    ## get the histogram
    image_hist = []
    for i in range(4):
        hist = torch.histc(image[i], bins=120, min=0, max=1)
        image_hist.append(hist)
    image_hist = torch.stack(image_hist)

    ## standardize the histogram
    n = image_hist.sum(dim=1, keepdim=True)
    image_hist_norm = (image_hist / n)[None, ...]

    ## Perform the inference
    with torch.no_grad():
        s2_hist_hat, _, _ = reflectance_model(image_hist_norm)
        s2_hist_hat = s2_hist_hat.squeeze()

    ## Correct the histogram based on the output of the VAE
    s2_hist_hat[s2_hist_hat < 0.005] = 0  # Remove small values
    s2_cumsum = s2_hist_hat.cumsum(dim=1)
    s2_cumsum_norm = s2_cumsum / s2_cumsum[:, -1:]

    image_cumsum = image_hist.cumsum(dim=1)
    image_cumsum_norm = image_cumsum / image_cumsum[:, -1:]

    # Simple rule NAIP reflectance always higher than S2
    s2_cumsum_norm[(image_cumsum_norm - s2_cumsum_norm) > 0] = 0
    s2_cumsum_norm = s2_cumsum_norm / s2_cumsum_norm[:, -1:]
    s2_cumsum_norm = s2_cumsum_norm.clamp(0, 1).cpu().numpy()

    # Generate a fake image with the corrected histogram (s2_cumsum_norm)
    random_values = np.random.rand(*image.shape)
    image_hat = torch.zeros_like(image)
    for i in range(4):
        for j in range(120):
            image_hat[i][random_values[i] > s2_cumsum_norm[i][j]] = j / 120

    # Copy the histogram of the original image
    return hq_histogram_matching(image, image_hat)


def harmonization(image: torch.Tensor, params: dict) -> torch.Tensor:
    
    # Set the parameters
    params = params.copy()
    methods = params.pop("reflectance_method", "gamma_multivariate_normal")
    
    # Check if the percentiles are a list
    percentiles = params.pop("percentiles", [50])
    if not isinstance(percentiles, list):
        percentiles = [percentiles]

    container = {}

    if "identity" in methods:
        device = params.get("device", "cpu")
        container["identity"] = image[None, ...].to(device)

    if "gamma_lognormal" in methods:
        con_perc = []
        for perc in percentiles:
            con_perc.append(
                harmonization_01(
                    image=image,
                    percentiles=perc,
                    **params
                )
            )
        container["gamma_lognormal"] = torch.stack(con_perc)

    if "gamma_multivariate_normal" in methods:
        con_perc = []
        for perc in percentiles:            
            con_perc.append(
                harmonization_02(
                    image=image, 
                    percentiles=perc,
                    **params
                )
            )
        container["gamma_multivariate_normal"] = torch.stack(con_perc)        

    if "unet_histogram_matching" in methods:
        container["unet_histogram_matching"] = harmonization_03(
            image=image, **params
        )[None, ...]
    
    if "vae_histogram_matching" in methods:
        container["vae_histogram_matching"] = harmonization_04(
            image=image, **params
        )[None, ...]
    
    if len(container) == 0:
        raise ValueError("No method was selected")
    
    # orderdict considering the order of the methods
    order_container = OrderedDict((method, container[method]) for method in methods)
    
    return torch.concatenate(list(order_container.values()), dim=0)