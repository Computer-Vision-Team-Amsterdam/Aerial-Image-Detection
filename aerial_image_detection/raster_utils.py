from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio
import shapely.geometry as sg
from rasterio.mask import mask
from rasterio.plot import reshape_as_image, show
from rasterio.transform import array_bounds


class RasterData:
    raster_img: Optional[npt.NDArray] = None

    def __init__(self, file_path: str):
        self.file_path = file_path

        with rasterio.open(file_path, "r") as src:
            self.raster_img = src.read()
            self.description_str = RasterData._get_description(src)
            self.bounds = src.bounds

    @classmethod
    def _get_description(cls, raster_data) -> str:
        full_width_m = raster_data.bounds.right - raster_data.bounds.left
        full_height_m = raster_data.bounds.top - raster_data.bounds.bottom
        cm_per_px = (
            (raster_data.bounds.right - raster_data.bounds.left)
            / raster_data.width
            * 100
        )

        description = (
            f"Image shape: {raster_data.shape} pixels\n"
            f" - height: {full_height_m}m\n"
            f" - width:  {full_width_m}m\n"
            f" - {cm_per_px} cm/pixel"
        )

        return description

    def description(self) -> str:
        return self.description_str

    def as_rgb_img(self) -> npt.NDArray:
        return reshape_as_image(self.raster_img)

    def as_bgr_img(self):
        return np.flip(self.as_rgb_img(), 2)

    def get_shapely_transform(self) -> List[float]:
        with rasterio.open(self.file_path, "r") as src:
            return src.transform.to_shapely()

    def get_bounds_as_polygon(self) -> sg.Polygon:
        return sg.box(*tuple(self.bounds))

    def get_crop(
        self, bounds: Tuple[float, float, float, float]
    ) -> Tuple[Optional[npt.NDArray], Optional[sg.Polygon]]:
        crop_poly = sg.box(*bounds)
        if not crop_poly.intersects(self.get_bounds_as_polygon()):
            return None, None

        with rasterio.open(self.file_path, "r") as src:
            out_img, out_transform = mask(dataset=src, shapes=[crop_poly], crop=True)
            out_poly = sg.box(
                *array_bounds(out_img.shape[1], out_img.shape[2], out_transform)
            )
            return reshape_as_image(out_img), out_poly

    def get_relative_crop(self, rel_bounds: Tuple[float, float, float, float]):
        bounds = (
            self.bounds.left + rel_bounds[0],
            self.bounds.bottom + rel_bounds[1],
            self.bounds.left + rel_bounds[2],
            self.bounds.bottom + rel_bounds[3],
        )
        return self.get_crop(bounds)

    def show(self):
        with rasterio.open(self.file_path, "r") as src:
            show(src)
