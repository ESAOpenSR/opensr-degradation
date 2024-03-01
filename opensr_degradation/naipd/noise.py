import pkg_resources
import numpy as np
import torch

from typing import Union

def noise(
    image: torch.Tensor,
    device: Union[str, torch.device] = "cpu",
    **kwargs
) -> torch.Tensor:
    """ Create noisy image from a noisy matrix

    Args:
        image (torch.Tensor): The image to be noised. It must
            be a numpy array or a torch tensor with shape (C, H, W).
        noisy_matrix (np.ndarray): The noisy matrix. It must be a numpy array.

    Returns:
        torch.Tensor: The noisy image.
    """
    
    # Load the noisy matrix
    file = pkg_resources.resource_filename(
        "opensr_degradation", "naipd/models/model_noise.pt"
    )
    noisy_matrix = torch.load(file).detach().cpu().numpy()
    
    # if image is torch tensor, convert to numpy
    image = image.detach().cpu().numpy()
    noisy_matrix = noisy_matrix

    # Ranges of the SNR matrix
    reflectance_ranges = np.arange(0, 0.5, 0.005)
    noisy_ranges = np.arange(-0.0101, 0.0101, 0.0002)

    # Categorize the reflectance
    r_cat = np.digitize(image, reflectance_ranges)

    # Create noisy model
    vfunc = np.vectorize(lambda x: np.random.choice(noisy_ranges, p=noisy_matrix[x,]))
    return torch.from_numpy(vfunc(r_cat)).squeeze().float().to(device)
