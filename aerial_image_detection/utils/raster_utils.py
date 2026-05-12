from typing import List, Optional, Tuple

import numpy as np
import numpy.typing as npt
import rasterio
import shapely.geometry as sg
from rasterio.io import DatasetReader
from rasterio.mask import mask
from rasterio.plot import reshape_as_image, show
from rasterio.transform import array_bounds


class RasterData:
    """
    Helper class to more easily deal with GeoTIFF images through
    :class:`~rasterio`. This class adds methods to crop a GeoTIFF raster to
    specified bounds, get the image as either GRB or BGR, and get its transform
    from pixels to real-world coordinates.

    This class is designed to only load the actual raster data when needed, e.g.
    when requesting the image. For other purposes, only the header is read.

    Parameters
    ----------
    file_path: str
        Path to the GeoTIFF file.
    """

    raster_img: Optional[npt.NDArray] = None

    def __init__(self, file_path: str):
        self.file_path = file_path

        with rasterio.open(file_path, "r") as src:
            self.description_str = RasterData._get_description(src)
            self.bounds = src.bounds

    @classmethod
    def _get_description(cls, raster_data: DatasetReader) -> str:
        """
        Get a description string for the given
        :class:`rasterio.io.DatasetReader` which includes its shape in pixels,
        dimensions in real-world meters, and the conversion rate between cm and
        pixels.
        """
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
        """
        Get a description string for this GeoTIFF file which includes its shape
        in pixels, dimensions in real-world meters, and the conversion rate
        between cm and pixels.
        """
        return self.description_str

    def as_rgb_img(self) -> npt.NDArray:
        """
        Get the GeoTIFF data as RGB image. This assumes the data is indeed RGB shaped.
        """
        if self.raster_img is None:
            with rasterio.open(self.file_path, "r") as src:
                self.raster_img = src.read()

        return reshape_as_image(self.raster_img)

    def as_bgr_img(self) -> npt.NDArray:
        """
        Get the GeoTIFF data as BGR image. This assumes the data is RGB shaped,
        and flips it accordingly.
        """
        return np.flip(self.as_rgb_img(), 2)

    def get_image(self) -> npt.NDArray:
        """
        Get the GeoTIFF data as (RGB) image. This assumes the data is indeed an image.
        """
        return self.as_rgb_img()

    def get_shapely_transform(self) -> List[float]:
        """
        Get the
        [shapely.affinity.affine_transform](https://shapely.readthedocs.io/en/stable/manual.html#shapely.affinity.affine_transform)
        compatible transform from pixels to real-world coordinates.
        """
        with rasterio.open(self.file_path, "r") as src:
            return src.transform.to_shapely()

    def get_bounds_as_polygon(self) -> sg.Polygon:
        """
        Get the real-world bounds of the GeoTIFF data as
        :class:`~shapely.sg.Polygon`.
        """
        return sg.box(*tuple(self.bounds))

    def get_crop(
        self, bounds: Tuple[float, float, float, float]
    ) -> Tuple[Optional[npt.NDArray], Optional[sg.Polygon], Optional[List[float]]]:
        """
        Crop the GeoTIFF image to the specified bounds.

        Parameters
        ----------
        bounds: Tuple[float, float, float, float]
            Real-world bounds as [left, bottom, right, top]

        Returns
        -------
        The cropped image as a tuple: (image data, bounds, transform).
        """
        crop_poly = sg.box(*bounds)
        if not crop_poly.intersects(self.get_bounds_as_polygon()):
            return None, None, None

        with rasterio.open(self.file_path, "r") as src:
            out_img, out_transform = mask(dataset=src, shapes=[crop_poly], crop=True)
            out_poly = sg.box(
                *array_bounds(out_img.shape[1], out_img.shape[2], out_transform)
            )
            return reshape_as_image(out_img), out_poly, out_transform.to_shapely()

    def get_relative_crop(
        self, rel_bounds: Tuple[float, float, float, float]
    ) -> Tuple[Optional[npt.NDArray], Optional[sg.Polygon], Optional[List[float]]]:
        """
        Crop the GeoTIFF image to the specified relative bounds in relation to
        this GeoTIFF's bounds.

        Parameters
        ----------
        rel_bounds: Tuple[float, float, float, float]
            Relative bounds as [left, bottom, right, top] w.r.t. the bottom-left
            corner of the GeoTIFF.

        Returns
        -------
        The cropped image as a tuple: (image data, bounds, transform).
        """
        bounds = (
            self.bounds.left + rel_bounds[0],
            self.bounds.bottom + rel_bounds[1],
            self.bounds.left + rel_bounds[2],
            self.bounds.bottom + rel_bounds[3],
        )
        return self.get_crop(bounds)

    def show(self):
        """Show the image."""
        with rasterio.open(self.file_path, "r") as src:
            show(src)
