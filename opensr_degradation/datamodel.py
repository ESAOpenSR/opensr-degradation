from typing import Any, List, Optional

import pydantic
import torch
import torchvision


class Sensor(pydantic.BaseModel):
    bands: List[str]
    sensor_name: str
    resolution: List[int]
    blur_model: Optional[Any] = None
    reflectance_model: Optional[Any] = None
    center_wavelength: Optional[List[float]] = None
    full_width_half_max: Optional[List[float]] = None

    @pydantic.model_validator(mode="after")
    def check_length(cls, v):
        cw_exists = False
        fwhm_exists = False
        res_exists = False

        if v.center_wavelength is not None:
            cw_exists = True
        if v.full_width_half_max is not None:
            fwhm_exists = True
        if v.resolution is not None:
            res_exists = True

        if cw_exists and fwhm_exists and res_exists:
            if (
                len(v.resolution)
                != len(v.center_wavelength)
                != len(v.full_width_half_max)
            ):
                raise ValueError(
                    "Resolution, center_wavelength, and full_width_half_max must be the same length"
                )

        if cw_exists and fwhm_exists and not res_exists:
            if len(v.center_wavelength) != len(v.full_width_half_max):
                raise ValueError(
                    "Center_wavelength and full_width_half_max must be the same length"
                )

        if cw_exists and not fwhm_exists and res_exists:
            if len(v.resolution) != len(v.center_wavelength):
                raise ValueError(
                    "Resolution and center_wavelength must be the same length"
                )

        if not cw_exists and fwhm_exists and res_exists:
            if len(v.resolution) != len(v.full_width_half_max):
                raise ValueError(
                    "Resolution and full_width_half_max must be the same length"
                )

        return v
