<p align="center">
  <img src=https://github.com/ESAOpenSR/opensr-degradation/assets/16768318/2d2283d8-dff4-4272-90c9-221ed922e744 width=50%>
</p>

<p align="center">
    <em>A set of methods to emulate Sentinel-2 based on High-Resolution (NAIP) imagery</em>
</p>

<p align="center">
<a href='https://pypi.python.org/pypi/opensr-test'>
    <img src='https://img.shields.io/pypi/v/opensr-test.svg' alt='PyPI' />
</a>

<a href="https://opensource.org/licenses/MIT" target="_blank">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</a>
<a href='https://opensr-test.readthedocs.io/en/latest/?badge=main'>
    <img src='https://readthedocs.org/projects/opensr-test/badge/?version=main' alt='Documentation Status' />
</a>
<a href="https://github.com/psf/black" target="_blank">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Black">
</a>
<a href="https://pycqa.github.io/isort/" target="_blank">
    <img src="https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336" alt="isort">
</a>
</p>

---

**GitHub**: [https://github.com/ESAOpenSR/opensr-degradation](https://github.com/ESAOpenSR/opensr-degradation)

**Documentation**: [https://esaopensr.github.io/opensr-degradation](https://esaopensr.github.io/opensr-degradation)

**PyPI**: [https://pypi.org/project/opensr-degradation/](https://pypi.org/project/opensr-degradation/)

**Paper**: Coming soon!

---

## Overview

We created **three distinct degradation** models capable of transforming $NAIP$ images to closely emulate the characteristics of 
Sentinel-2 imagery, resulting in what we refer to as S2-like images.

**statistical**:  We identify the best gamma correction for each band pair between NAIP and Sentinel-2 images. Following this, we employ a 
multivariate Gaussian Distribution to learn the gamma distribution. During inference, the harmonized $NAIP$, $NAIP_{\hat{h}}$, is obtained by applying 
the four-dimensional gamma correction vector to the original $NAIP$ RGBNIR image.

**deterministic**: We train a U-Net architecture with EfficientNet-B0 as its backbone. The input data consists of NAIP imagery 
degraded to a 10-meter ($NAIP_{10m}$) resolution using simple bilinear interpolation with an anti-aliasing filter. The target data
comprises Sentinel-2 imagery. During inference, the harmonized $NAIP$, $NAIP_{\hat{h}}$, is obtained by a three-step process. First, 
we obtain $NAIP_{10m}$. Second, we use the U-Net model to infer an initial harmonized version of $NAIP$. Third, we use the U-Net 
prediction as a reference to correct the initial $NAIP$ reflectance values via histogram matching.

**variational**: We disaggregate each band from Sentinel-2 and $NAIP_{10m}$ into a 1D Tensor containing the number of values inside 
a histogram bin. Each histogram was structured into 120 bins from 0 to 1, transforming an image pair into two tensors with dimensions 
of (4, 100). Then, we use this transformed version of the dataset to train a variational autoencoder (VAE) that learns to transform 
the histogram of NAIP into the histogram of Sentinel-2. During inference, the harmonized NAIP, $NAIP_{\hat{h}}$, is obtained by a 
four-step process. First, we obtain $NAIP_{10m}$. Second, we obtain the histograms for each band. Third, we use the trained VAE 
to obtain the Sentinel-2 histogram prediction. Fourth, correct the initial NAIP reflectance values using the VAE predictions.

## How to use

The example below shows how to use `opensr-degradation` to convert a $NAIP$ to $NAIP_{\hat{h}}$ and $S2_{like}$. Using 
[cubo](https://github.com/ESDS-Leipzig/cubo), we can run `opensr-degradation` anywhere!

```python
import ee
import cubo
import torch
import einops
import pathlib
import rioxarray
import numpy as np
import pandas as pd
import opensr_degradation

ee.Initialize() 

demo = cubo.create(
    lat=40.5389819819361, # Central latitude of the cube
    lon=-111.9839859008789, # Central longitude of the cube
    collection="USDA/NAIP/DOQQ", # Name of the STAC collection
    bands=["R", "G", "B", "N"], # Bands to include in the cube
    start_date="2010-01-01", # Start date of the cube
    end_date="2018-01-01", # End date of the cube
    edge_size=256, # Edge size of the cube (px)
    resolution=2.5, # Pixel size of the cube (m)
    gee=True
)[0].to_numpy() / 255

degradation_model = opensr_degradation.pipe(
    sensor="naip_d",
    add_noise=True,
    params={
        "reflectance_method": [
            "identity",
            "gamma_lognormal",
            "gamma_multivariate_normal",
            "unet_histogram_matching",
            "vae_histogram_matching",
        ],
        "noise_method": "gaussian_noise",
        "device": "cuda",
        "seed": 42,
        "percentiles": [50],
        "vae_reflectance_model": opensr_degradation.naip_vae_model("cuda"),
        "unet_reflectance_model": opensr_degradation.naip_unet_model("cuda"),
    },
)

# Load a NAIP imagery
naip_image = rioxarray.open_rasterio(
    "https://huggingface.co/datasets/isp-uv-es/SEN2NAIP/resolve/main/demo/cross-sensor/ROI_0000/hr.tif"
)
demo
# Run the model
lr, hr = degradation_model.forward(torch.from_numpy(demo).float())

```

## Installation

Install the latest version from PyPI:

```
pip install opensr-degradation
```

Upgrade `opensr-test` by running:

```
pip install -U opensr-degradation
```

Install the latest dev version from GitHub by running:

```
pip install git+https://github.com/ESAOpenSR/opensr-degradation
```


### SEN2NAIP

If you are using [SEN2NAIP](https://huggingface.co/datasets/isp-uv-es/SEN2NAIP), histograms have been computed and stored within the metadata file so that users can 
obtain the LR-HR pairs quickly. Use the `get_s2like` function for this.


```python
import opensr_degradation
import rioxarray
import datasets
import requests
import tempfile
import torch
import json


def load_metadata(metadata_path: str) -> dict:
    tmpfile = tempfile.NamedTemporaryFile(suffix=".json")
    with requests.get(metadata_path) as response:
        with open(tmpfile.name, "wb") as file:
            file.write(response.content)
        metadata_json = json.load(open(tmpfile.name, "r"))
    return metadata_json

DEMO_PATH = "https://huggingface.co/datasets/isp-uv-es/SEN2NAIP/resolve/main/demo/"

# Synthetic LR and HR data ------------------------------
synthetic_path = DEMO_PATH + "synthetic/ROI_0001/"

hr_early_data = rioxarray.open_rasterio(synthetic_path + "early/01__m_4506807_nw_19_1_20110818.tif")
hr_early_torch = torch.from_numpy(hr_early_data.to_numpy()) / 255
hr_early_metadata = load_metadata(synthetic_path + "late/metadata.json")
lr_hat, hr_hat = opensr_degradation.main.get_s2like(
    image=hr_early_torch,
    table=hr_early_metadata["sim_histograms"],
    model="gamma_multivariate_normal_50"
)


import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(hr_early_torch[[3, 1, 2]].permute(1, 2, 0))
ax[0].set_title("NAIP")
ax[1].imshow(hr_hat[[3, 1, 2]].permute(1, 2, 0)*3)
ax[1].set_title("NAIPhat")
ax[2].imshow(lr_hat[[3, 1, 2]].permute(1, 2, 0)*3)
ax[2].set_title("S2like")
plt.show()
```

<p align="center">
  <img src=https://github.com/ESAOpenSR/opensr-degradation/assets/16768318/c88fa16e-bbe7-4072-b518-5ab3b7278893 width=100%>
</p>
