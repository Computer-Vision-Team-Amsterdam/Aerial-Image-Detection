from typing import Dict, Optional

import geopandas as gpd
from shapely.geometry import Polygon

from aerial_image_detection.inference import SAHIModel
from aerial_image_detection.utils import CityAreaHandler


class AerialImageInference:

    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_settings: Dict,
    ):
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.settings = inference_settings

        self.sahi_model = self._setup_inference_model(model_path=model_path)
        self.target_area = self._parse_target_area()
        self.image_gdf = self._generate_image_gdf()

        # TODO
        pass

    def run(self):
        # TODO
        pass

    def _setup_inference_model(self, model_path: str) -> SAHIModel:
        sahi_model = SAHIModel(
            yolo_model_weights_path=model_path,
            confidence_treshold=self.settings["model_params"]["conf"],
            image_size=self.settings["model_params"]["img_size"],
            device=self.settings["model_params"]["device"],
            slice_height=self.settings["sahi_params"]["slice_height"],
            slice_width=self.settings["sahi_params"]["slice_width"],
            classes_to_keep=self.settings["target_classes"],
            class_agnostic=self.settings["sahi_params"]["postprocess_class_agnostic"],
        )

        return sahi_model

    def _parse_target_area(self) -> Optional[Polygon]:
        city_area_handler = CityAreaHandler()
        city_areas = city_area_handler.get_city_area_gdf()

        target_shape = None

        for key, value in self.settings["inference"]["target_area"].items():
            key_shape = city_areas[
                city_areas[f"{key}_naam"].str.lower() == value.lower()
            ].union_all()
            if target_shape is None:
                target_shape = key_shape
            else:
                target_shape = target_shape.union_all(key_shape)

        return target_shape

    def _generate_image_gdf(self) -> gpd.GeoDataFrame:
        pass
