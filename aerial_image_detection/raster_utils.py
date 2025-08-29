from typing import List, Optional

import numpy as np
import numpy.typing as npt
import rasterio
import shapely.geometry as sg
from rasterio.plot import reshape_as_image, show


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
        bounds_poly = sg.box(
            minx=self.bounds.left,
            miny=self.bounds.bottom,
            maxx=self.bounds.right,
            maxy=self.bounds.top,
        )
        return bounds_poly

    def show(self):
        with rasterio.open(self.file_path, "r") as src:
            show(src)
