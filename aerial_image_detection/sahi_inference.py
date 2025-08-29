from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from sahi import AutoDetectionModel
from sahi.predict import PredictionResult, get_sliced_prediction

ALL_CLASSES = {
    0: "plane",
    1: "ship",
    2: "storage tank",
    3: "baseball diamond",
    4: "tennis court",
    5: "basketball court",
    6: "ground track field",
    7: "harbor",
    8: "bridge",
    9: "large vehicle",
    10: "small vehicle",
    11: "helicopter",
    12: "roundabout",
    13: "soccer ball field",
    14: "swimming pool",
}


class SAHIInferenceModel:

    def __init__(
        self,
        yolo_model_weights_path: str,
        confidenc_treshold: float = 0.3,
        image_size: int = 1024,
        device: str = "cpu",
        slice_height: Optional[int] = None,
        slice_width: Optional[int] = None,
        overlap_height_ratio: float = 0.1,
        overlap_width_ratio: float = 0.1,
        classes_to_keep: Optional[List[int]] = None,
        class_agnostic: bool = False,
    ):
        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=yolo_model_weights_path,  # any yolov8/yolov9/yolo11/yolo12/rt-detr det model is supported  # noqa: E501
            confidence_threshold=confidenc_treshold,
            device=device,  # or 'cuda:0' if GPU is available
            image_size=image_size,
        )

        self.slice_height = slice_height or image_size
        self.slice_width = slice_width or image_size
        self.overlap_height_ratio = overlap_height_ratio
        self.ooverlap_width_ratio = overlap_width_ratio

        if classes_to_keep is not None:
            self.classes_to_exclude = [
                class_id
                for class_id in ALL_CLASSES.keys()
                if class_id not in classes_to_keep
            ]
        else:
            self.classes_to_exclude = None

        self.class_agnostic = class_agnostic

    def predict(self, image: npt.NDArray) -> PredictionResult:
        self.last_result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.ooverlap_width_ratio,
            exclude_classes_by_id=self.classes_to_exclude,
            postprocess_class_agnostic=self.class_agnostic,
        )
        return self.last_result

    def get_obb_cls_and_boxes(
        self,
        result: Optional[PredictionResult] = None,
    ) -> Tuple[List, List]:
        result = result or self.last_result

        obb_boxes = np.concatenate(
            [
                [
                    np.reshape(pred.mask.segmentation, [4, 2])
                    for pred in result.object_prediction_list
                ]
            ]
        ).tolist()
        obb_cls = np.array(
            [int(pred.category.id) for pred in result.object_prediction_list]
        ).tolist()
        return (obb_cls, obb_boxes)

    def get_names(self) -> Dict[int, str]:
        return self.detection_model.names
