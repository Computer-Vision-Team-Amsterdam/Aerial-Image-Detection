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

IMG_FORMATS = ".tif"


class Row(NamedTuple):
    filename: str
    geometry: Polygon
    in_target_area: bool
    overlap_bounds: Polygon
    is_cropped: bool


class AerialImageInference:

    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_settings: Dict,
    ):
        self.run_timestamp = datetime.now().strftime(format="%Y-%m-%dT%H-%M-%S")
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
        return detections_file

    def _process_row(self, detections_output_folder: str, row: Row) -> str:
        full_image_path = os.path.join(self.images_folder, row.filename)
        raster_data = RasterData(full_image_path)
        if row.is_cropped:
            image, _, transform = raster_data.get_crop(row.overlap_bounds.bounds)
        else:
            image = raster_data.get_image()
            transform = raster_data.get_shapely_transform()

        sahi_result = self.sahi_model.predict(image=image)

        total_time = sum(sahi_result.durations_in_seconds.values())
        time_string = ", ".join(
            [
                f"{key}: {value:.2f}"
                for key, value in sahi_result.durations_in_seconds.items()
            ]
        )
        logger.debug(f"Inference took {total_time:.2f} seconds ({time_string}).")

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
            f"{os.path.splitext(row.filename)[0]}.geojson",
        )
        detections_gdf.to_file(output_file_path, driver="GeoJSON")
        return output_file_path

    def _merge_and_write_detections(
        self, detections_output_folder: str, detections_file_list: List[str]
    ) -> str:
        full_detections_gdf = pd.concat(
            [gpd.read_file(detections_file) for detections_file in detections_file_list]
        )

        target_area_detections = full_detections_gdf[
            full_detections_gdf.intersects(self.target_polygon)
        ]
        target_area_detections.reset_index(inplace=True, drop=True)

        output_file_path = os.path.join(
            detections_output_folder, "combined_detections_in_target_area.geojson"
        )
        target_area_detections.to_file(output_file_path, driver="GeoJSON", index=True)
        return output_file_path

    def _setup_inference_model(self, model_path: str) -> SAHIModel:
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
        logger.debug("Parsing target area from config.yml...")
        logger.debug(self.settings["target_area"])

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
        PLOT_CRS = "EPSG:3857"

        target_shape_gdf = gpd.GeoDataFrame(
            data={"geometry": [self.target_polygon]}, crs=RD_CRS
        )
        fig, ax = plt.subplots(figsize=(15, 15))
        self.images_gdf.to_crs(PLOT_CRS).plot(
            ax=ax, alpha=0.5, column="in_target_area", edgecolor="darkblue"
        )
        self.images_gdf.set_geometry("overlap_bounds", crs=RD_CRS).to_crs(
            PLOT_CRS
        ).boundary.plot(ax=ax, edgecolor="red")
        target_shape_gdf.to_crs(PLOT_CRS).boundary.plot(ax=ax, edgecolor="black")
        cx.add_basemap(ax)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(file_path, bbox_inches=extent, dpi=150)

    def _generate_image_gdf(self) -> gpd.GeoDataFrame:
        def _bounds_row_to_poly(row: pd.Series) -> Optional[Polygon]:
            if not row.isna().all():
                return box(row.minx, row.miny, row.maxx, row.maxy)
            else:
                return None

        logger.debug(f"Analyzing and filtering input images in {self.images_folder}...")

        images = sorted(
            [
                filename
                for filename in os.listdir(self.images_folder)
                if any(filename.endswith(ext) for ext in IMG_FORMATS)
            ]
        )
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
        image_gdf["in_target_area"] = image_gdf.intersects(
            self.target_polygon.buffer(self.settings["target_area_buffer"])
        )
        image_gdf["overlap_bounds"] = [
            _bounds_row_to_poly(row)
            for _, row in (
                image_gdf.intersection(
                    self.target_polygon.buffer(self.settings["target_area_buffer"])
                ).bounds.iterrows()
            )
        ]
        image_gdf["is_cropped"] = [
            not a.equals(b)
            for a, b in zip(image_gdf["geometry"], image_gdf["overlap_bounds"])
        ]

        logger.debug(
            f"Found {len(images)} images in input folder, "
            f"of which {sum(image_gdf["in_target_area"])} intersect with target area."
        )

        return image_gdf
