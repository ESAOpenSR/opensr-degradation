from typing import Optional, Union

import torch
import torchvision

BLUR_MODEL = [
    torchvision.transforms.GaussianBlur(13, sigma=3.1),
    torchvision.transforms.GaussianBlur(13, sigma=3.1),
    torchvision.transforms.GaussianBlur(13, sigma=3.1),
    torchvision.transforms.GaussianBlur(13, sigma=3.1),
]


def blur(image: torch.Tensor, params: dict) -> torch.Tensor:
    """Apply a Blur kernel to an image tensor.

    Args:
        image (torch.Tensor): An image tensor.
        sensor (Sensor): A sensor object with a blur_model attribute.

    Returns:
        torch.Tensor: The blurred image tensor.
    """
    container = []
    for i in range(len(BLUR_MODEL)):
        container.append(BLUR_MODEL[i](image[i][None]))

    return torch.nn.functional.interpolate(
        torch.cat(container)[None], scale_factor=1 / 4, mode="bilinear", antialias=False
    ).squeeze()
