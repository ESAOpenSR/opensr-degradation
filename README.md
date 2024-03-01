# opensr-degradation

<div align="center">

</div>

## Install

```python
pip install opensr-degradation
```

## Usage

```python
import opensr_degradation
import torch

degradation_model = opensr_degradation.pipe(
    sensor="naip_d",
    add_noise=True,
    params={
        "method": [
            "identity",
            "gamma_lognormal",
            "gamma_multivariate_normal",
            "unet_histogram_matching",
            "vae_histogram_matching",
        ],
        "device": "cuda",
        "seed": 42,
        "percentiles": [10, 25, 50, 75, 90],
    },
)

image = torch.rand(4, 256, 256)
lr, hr = degradation_model(image)
``` 
