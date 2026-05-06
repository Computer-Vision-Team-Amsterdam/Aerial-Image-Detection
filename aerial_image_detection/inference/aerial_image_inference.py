import os
from datetime import datetime
from typing import Dict, List, NamedTuple, Optional

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon, box

from aerial_image_detection import logger
from aerial_image_detection.constants import RD_CRS
from aerial_image_detection.inference.sahi_model import SAHIModel
from aerial_image_detection.utils.city_area_handler import CityAreaHandler
from aerial_image_detection.utils.raster_utils import RasterData

IMG_FORMATS = [".tif"]  # Default image file format, can be overwritten in settings


class Row(NamedTuple):
    """
    One row of the generated images GeoDataFrame.

    Attributes
    ----------
    filename: str
        The name of the input file.
    geometry: Polygon
        The geometry of the input file in real-world coordinates.
    in_target_area: bool
        Whether or not the geometry overlaps with the target area.
    overlap_bounds: Polygon
        The rectangular bounds of the overlap area, used for cropping.
    is_cropped: bool
        Whether or not the input file can be cropped (i.e. when the overlap is smaller than the input geometry).
    """

    filename: str
    geometry: Polygon
    in_target_area: bool
    overlap_bounds: Polygon
    is_cropped: bool


class AerialImageInference:
    """
    Perform inference on aerial photography.

    The current version expects a [YOLO OBB
    model](https://docs.ultralytics.com/tasks/obb/) that can be used with
    [SAHI](https://github.com/obss/sahi). Input images are assumed to be in
    GeoTIFF format, for example such as can be downloaded from
    https://www.beeldmateriaal.nl/.

    The constructor analyses the given input folder and target area as specified
    in the settings, and produces a map view of the input data cropped to this
    target area.

    Parameters
    ----------
    images_folder: str
        Path to input images (expected to be in GeoTIFF format).
    output_folder: str
        Path to folder where output (map plot and results) will be written.
    model_path: str
        Path to YOLO OBB model weights.
    inference_settings: Dict
        Settings (as parsed from config.yml). See `InferenceSpec` in
        aerial_image_detection/settings/settings_schema for details.
    """

    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_settings: Dict,
    ):
        self.run_timestamp = datetime.now().strftime(format="%Y-%m-%dT%H-%M-%S")  # type: ignore
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.settings = inference_settings

        self.sahi_model = self._setup_inference_model(model_path=model_path)
        self.target_polygon = self._parse_target_area()
        self.images_gdf = self._generate_image_gdf()

        plot_file_path = os.path.join(
            self.output_folder, f"input_analysis_plot_{self.run_timestamp}.png"
        )
        self._plot_image_gdf(file_path=plot_file_path)
        logger.info(f"Input folder analysis plotted and saved to {plot_file_path}")

    def run(self) -> str:
        """
        Run the object detection. This will process each input image that
        overlaps the target area, pass it through the SAHI sliced inference, and
        store the results. Intermediate result files will be written for each
        input image; finally these will be merged and intermediate results will
        be removed.
        """
        detections_output_folder = os.path.join(
            self.output_folder, f"detections_{self.run_timestamp}"
        )
        os.makedirs(detections_output_folder, exist_ok=True)

        detections_file_list = []
        images_to_do = self.images_gdf[self.images_gdf["in_target_area"]]

        for i, row in enumerate(images_to_do.itertuples()):
            logger.info(f"Processing file {i + 1}/{len(images_to_do)} : {row.filename}")

            output_file_path = self._process_row(detections_output_folder, row)

            logger.debug(f"Predictions written to {output_file_path}")

            detections_file_list.append(output_file_path)

        detections_file = self._merge_and_write_detections(
            detections_output_folder, detections_file_list
        )
        logger.info(f"Detections merged and written to {detections_file}")

        logger.info("Cleaning up temporary files...")
        self._cleanup_detection_files(detections_file_list)

        return detections_file

    def _process_row(self, detections_output_folder: str, row: Row) -> str:
        """
        Process one image row in the images GDF generated in the constructor.

        Parameters
        ----------
        detections_output_folder: str
            Path to folder where results will be stored.
        row: Row
            One Row from the images_gdf.

        Returns
        -------
        Path to the generated results file.
        """
        full_image_path = os.path.join(self.images_folder, row.filename)

        # This assumes the input image is GeoTIFF
        raster_data = RasterData(full_image_path)
        if row.is_cropped:
            image, _, transform = raster_data.get_crop(row.overlap_bounds.bounds)
        else:
            image = raster_data.get_image()
            transform = raster_data.get_shapely_transform()

        n_slices = self.sahi_model.get_number_of_slices_for_image(image=image)
        logger.debug(f"Performing prediction on {n_slices} slices.")

        # Run the object detection
        sahi_result = self.sahi_model.predict(image=image)

        total_time = sum(sahi_result.durations_in_seconds.values())
        time_string = ", ".join(
            [
                f"{key}: {value:.2f}"
                for key, value in sahi_result.durations_in_seconds.items()
            ]
        )
        logger.debug(f"Inference took {total_time:.2f} seconds ({time_string}).")

        # Convert bounding boxes from image pixels to real world coordinates in the GeoTIFF image
        sahi_predictions = self.sahi_model.get_prediction_data()
        sahi_predictions["geometry"] = gpd.GeoSeries(
            data=[Polygon(coords) for coords in sahi_predictions["bounding_box"]]
        ).affine_transform(transform)

        detections_gdf = gpd.GeoDataFrame(
            data=sahi_predictions,
            crs=RD_CRS,
        )
        detections_gdf.insert(0, column="source_file", value=row.filename)

        output_file_path = os.path.join(
            detections_output_folder,
            f"{os.path.splitext(row.filename)[0]}.gpkg",
        )
        detections_gdf.to_file(output_file_path, driver="GPKG")
        return output_file_path

    def _merge_and_write_detections(
        self, detections_output_folder: str, detections_file_list: List[str]
    ) -> str:
        """
        Merge intermediate results files into one final output file.

        Parameters
        ----------
        detections_output_folder: str
            Path to folder where results will be stored.
        detections_file_list: List[str]
            List of paths of files to be merged.

        Returns
        -------
        Path to the merged results file.
        """
        full_detections_gdf = pd.concat(
            [gpd.read_file(detections_file) for detections_file in detections_file_list]
        )

        target_area_detections = full_detections_gdf[
            full_detections_gdf.intersects(self.target_polygon)
        ]
        target_area_detections.reset_index(inplace=True, drop=True)

        output_file_path = os.path.join(
            detections_output_folder, "combined_detections_in_target_area.gpkg"
        )
        target_area_detections.to_file(output_file_path, driver="GPKG", index=True)
        return output_file_path

    def _cleanup_detection_files(self, detections_file_list: List[str]) -> None:
        """
        Remove intermediate results files (after merging).

        Parameters
        ----------
        detections_file_list: List[str]
            List of paths of files to be deleted.
        """
        for detection_file in detections_file_list:
            try:
                os.remove(detection_file)
            except Exception as e:
                logger.error(f"Failed to remove file '{detection_file}': {str(e)}")
                raise Exception(f"Failed to remove file '{detection_file}': {e}")

    def _setup_inference_model(self, model_path: str) -> SAHIModel:
        """
        Set-up the SAHI inference model using the configuration options
        specified in the settings.
        """
        logger.info(f"Loading SAHI model at {model_path}...")
        return SAHIModel(
            yolo_model_weights_path=model_path,
            confidence_treshold=self.settings["model_params"]["conf"],
            image_size=self.settings["model_params"]["img_size"],
            device=self.settings["model_params"]["device"],
            slice_height=self.settings["sahi_params"]["slice_height"],
            slice_width=self.settings["sahi_params"]["slice_width"],
            classes_to_keep=self.settings["target_classes"],
            class_agnostic=self.settings["sahi_params"]["postprocess_class_agnostic"],
        )

    def _parse_target_area(self) -> Optional[Polygon]:
        """Convert the specified target area into a polygon. The target area is
        assumed to be a combination of city part and name, e.g. `'stadsdeel':
        'Centrum'` or `'wijk': 'Nieuwmarkt/Lastage'`."""
        logger.debug("Parsing target area from config.yml...")
        logger.debug(self.settings["target_area"])

        # Get the city area geometries from the API
        city_area_handler = CityAreaHandler()
        city_areas = city_area_handler.get_city_area_gdf()

        target_shape = None

        for key, value in self.settings["target_area"].items():
            key_shape = city_areas[
                city_areas[f"{key}_naam"].str.lower() == value.lower()
            ].union_all()
            if target_shape is None:
                target_shape = key_shape
            else:
                target_shape = target_shape.union_all(key_shape)

        return target_shape

    def _plot_image_gdf(self, file_path: str) -> None:
        """
        Plot the input images overlaid on the target area, and visualize how each will be cropped.

        Parameters
        ----------
        file_path: str
            Path where to store the image plot.
        """
        PLOT_CRS = "EPSG:3857"  # LatLon coordinates
        TILE_URL = "https://t1.data.amsterdam.nl/topo_wm/{z}/{x}/{y}.png"  # Amsterdam tile server

        # Convert to GDF for easy plotting
        target_shape_gdf = gpd.GeoDataFrame(
            data={"geometry": [self.target_polygon]}, crs=RD_CRS
        )

        fig, ax = plt.subplots(figsize=(15, 15))

        # Plot input images GDF
        self.images_gdf.to_crs(PLOT_CRS).plot(
            ax=ax, alpha=0.5, column="in_target_area", edgecolor="darkblue"
        )
        # Plot overlap between images and target area
        self.images_gdf.set_geometry("overlap_bounds", crs=RD_CRS).to_crs(
            PLOT_CRS
        ).boundary.plot(ax=ax, edgecolor="red")
        # Plot target area
        target_shape_gdf.to_crs(PLOT_CRS).boundary.plot(ax=ax, edgecolor="black")
        # Add background street map
        cx.add_basemap(ax, source=TILE_URL)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(file_path, bbox_inches=extent, dpi=150)

    def _generate_image_gdf(self) -> gpd.GeoDataFrame:
        """
        Generate a GeoDataFrame with all input images, and enrich these with each image's overlap with the target area.

        Returns
        -------
        The GeoDataFrame with one :class:`~Row` for each input image..
        """

        def _bounds_row_to_poly(row: pd.Series) -> Optional[Polygon]:
            if not row.isna().all():
                return box(row.minx, row.miny, row.maxx, row.maxy)
            else:
                return None

        logger.debug(f"Analyzing and filtering input images in {self.images_folder}...")

        allowed_suffixes = self.settings["input_folder"].get(
            "allowed_suffixes", IMG_FORMATS
        )
        if isinstance(allowed_suffixes, str):
            allowed_suffixes = [allowed_suffixes]
        logger.debug(f"Allowed suffixes: {allowed_suffixes}")

        # Get list of input files
        images = sorted(
            [
                filename
                for filename in os.listdir(self.images_folder)
                if any(filename.endswith(ext) for ext in allowed_suffixes)
            ]
        )
        # Add the name and geometry of each input file to the GeoDataFrame
        image_gdf = gpd.GeoDataFrame(
            data={
                "filename": images,
                "geometry": [
                    RasterData(
                        os.path.join(self.images_folder, img)
                    ).get_bounds_as_polygon()
                    for img in images
                ],
            },
            crs=RD_CRS,
        )
        # Check overlap with target area
        image_gdf["in_target_area"] = image_gdf.intersects(
            self.target_polygon.buffer(self.settings["target_area_buffer"])
        )
        # Compute overlap bounds for cropping
        image_gdf["overlap_bounds"] = [
            _bounds_row_to_poly(row)
            for _, row in (
                image_gdf.intersection(
                    self.target_polygon.buffer(self.settings["target_area_buffer"])
                ).bounds.iterrows()
            )
        ]
        # Image can be cropped if overlap bounds are not equal to the geometry
        image_gdf["is_cropped"] = [
            not a.equals(b)
            for a, b in zip(image_gdf["geometry"], image_gdf["overlap_bounds"])
        ]

        logger.debug(
            f"Found {len(images)} images in input folder, "
            f"of which {sum(image_gdf['in_target_area'])} intersect with target area."
        )

        return image_gdf
