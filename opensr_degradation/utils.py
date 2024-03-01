import torch
import numpy as np

from skimage.exposure import match_histograms


def hq_histogram_matching(image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    """Lazy implementation of histogram matching

    Args:
        image1 (torch.Tensor): The low-resolution image (C, H, W).
        image2 (torch.Tensor): The super-resolved image (C, H, W).

    Returns:
        torch.Tensor: The super-resolved image with the histogram of
            the target image.
    """

    # Go to numpy
    np_image1 = image1.detach().cpu().numpy()
    np_image2 = image2.detach().cpu().numpy()

    if np_image1.ndim == 4:
        container = []
        for i in range(np_image1.shape[0]):
            container.append(
                match_histograms(np_image1[i], np_image2[i], channel_axis=0)
            )
        np_image1_hat = np.stack(container)
    elif np_image1.ndim == 3:
        np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=0)
    elif np_image1.ndim == 2:
        np_image1_hat = match_histograms(np_image1, np_image2, channel_axis=None)
    else:
        raise ValueError("The input image must have 2 or 3 dimensions.")

    # Go back to torch
    image1_hat = torch.from_numpy(np_image1_hat).to(image1.device)

    return image1_hat