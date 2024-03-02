import pkg_resources
import numpy as np
import torch

from typing import Union

def create_noisy_1(image: torch.Tensor) -> torch.Tensor:
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
    return torch.from_numpy(vfunc(r_cat)).squeeze().float()


def create_noisy_2(img: torch.Tensor) -> torch.Tensor:
    """ Create noisy image from a noisy matrix

    Args:
        img (torch.Tensor): The image to be noised. It must
            be a torch tensor with shape (C, H, W).
    Returns:
        torch.Tensor: The noisy image.
    """
    rnoisy = torch.normal(0, 0.025, size=img.shape).to(img.device)
    ratio_noisy = torch.sqrt(torch.mean(img ** 2, dim=(1, 2), keepdim=True)) * rnoisy
    return ratio_noisy


def noise(
    img: torch.Tensor,
    params: dict,
) -> torch.Tensor:
    """ Create noisy image from a noisy matrix

    Args:
        img (torch.Tensor): The image to be noised. It must
            be a torch tensor with shape (C, H, W).

    Returns:
        torch.Tensor: The noisy image.
    """

    # Create noisy image
    method = params.get("noise_method", "real_noise")
    device = params.get("device", "cpu")    

    if method == "no_noise":
        tensor_noise = torch.zeros_like(img)
    elif method == "real_noise":
        container = []
        for i in range(img.shape[0]):
            container.append(
                create_noisy_1(img[i])
            )
        tensor_noise = torch.stack(container)
    elif method == "gaussian_noise":
        container = []
        for i in range(img.shape[0]):
            container.append(
                create_noisy_2(img[i])
            )
        tensor_noise = torch.stack(container)
    else:
        raise ValueError("The method is not valid")

    return tensor_noise.to(device)