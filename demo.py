import torch
import pathlib
import rioxarray
import numpy as np
import opensr_degradation

PATH_1 = pathlib.Path("/home/cesar/Documents/paper_julio/cross-sensor/cross-sensor/")
rois = list(PATH_1.glob("*"))

_, x1, x2 = list(rois[1100].glob("*"))


s2_img = rioxarray.open_rasterio(x2)
img1 = rioxarray.open_rasterio(x1)
img1 = torch.from_numpy(img1.to_numpy()) / 255


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

image = img1
lr, hr = degradation_model(image)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 2, figsize=(10, 5))
for i in range(4):
    ax[i, 0].imshow(lr[i, 0:3].permute(1, 2, 0).cpu().numpy())
    ax[i, 1].imshow(hr[i, 0:3].permute(1, 2, 0).cpu().numpy())
# remove the x and y axis
for i in range(4):
    for j in range(2):
        ax[i, j].axis("off")
plt.show()




read_dist = s2_img.to_numpy().mean(axis=0)/10000
lr_dist = lr.cpu().numpy().mean(axis=0)

read_dist = read_dist.flatten()
lr_dist = lr_dist.flatten()


lr_rgb = np.transpose(lr.cpu().numpy()[0:3], (1, 2, 0))
hr_rgb = np.transpose(hr.cpu().numpy()[0:3], (1, 2, 0))
naip_rgb = np.transpose(img1.cpu().numpy()[0:3], (1, 2, 0))

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(10, 5))
ax[0].imshow(naip_rgb)
ax[1].imshow(lr_rgb*3)
ax[2].imshow(hr_rgb*3)
plt.show()

# load a package to pypi using poetry
# poetry build
# poetry publish


# publish to pypi
#!twine upload dist/*

lr_01, hr_01 = degradation_model(naip_torch)
lr_02, hr_02 = degradation_model(naip_torch)
lr_03, hr_03 = degradation_model(naip_torch)
lr_04, hr_04 = degradation_model(naip_torch)

naip_torch[0:3].permute(1, 2, 0).shape
import matplotlib.pyplot as plt
fig, ax = plt.subplots(4, 4, figsize=(10, 5))
ax[0, 0].imshow(naip_rgb)
ax[0, 1].imshow(lr_01.cpu().numpy().transpose(1, 2, 0))
ax[0, 2].imshow(hr_01.cpu().numpy().transpose(1, 2, 0))
ax[0, 3].hist(lr_01.cpu().numpy().flatten(), bins=100)
#add title
ax[0, 0].set_title("NAIP")
ax[1, 0].imshow(naip_rgb)
ax[1, 1].imshow(lr_02.cpu().numpy().transpose(1, 2, 0))
ax[1, 2].imshow(hr_02.cpu().numpy().transpose(1, 2, 0))
ax[1, 3].hist(lr_02.cpu().numpy().flatten(), bins=100)
ax[2, 0].imshow(naip_rgb)
ax[2, 1].imshow(lr_03.cpu().numpy().transpose(1, 2, 0))
ax[2, 2].imshow(hr_03.cpu().numpy().transpose(1, 2, 0))
ax[2, 3].hist(lr_03.cpu().numpy().flatten(), bins=100)
ax[3, 0].imshow(naip_rgb)
ax[3, 1].imshow(lr_04.cpu().numpy().transpose(1, 2, 0))
ax[3, 2].imshow(hr_04.cpu().numpy().transpose(1, 2, 0))
ax[3, 3].hist(lr_04.cpu().numpy().flatten(), bins=100)
# remove the x and y axis
for i in range(4):
    for j in range(4):
        ax[i, j].axis("off")

plt.show()

