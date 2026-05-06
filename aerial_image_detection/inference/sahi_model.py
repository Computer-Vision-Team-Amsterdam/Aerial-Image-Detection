from typing import Dict, List, Optional

import numpy as np
import numpy.typing as npt
from sahi import AutoDetectionModel
from sahi.predict import PredictionResult, get_sliced_prediction
from sahi.slicing import get_slice_bboxes

from aerial_image_detection.constants import OBB_CLASSES


class SAHIModel:
    """
    Wrapper around [SAHI](https://github.com/obss/sahi) for convenience.

    Can be used to run sliced prediction and convert the results to [YOLO OBB
    model](https://docs.ultralytics.com/tasks/obb/) bounding boxes.

    Usage
    -----
    ```python
    model = SAHIModel(...)
    model.predict(image=...)
    results = model.get_prediction_data()
    ```

    Parameters
    ----------
    yolo_model_weights_path: str
        Path to YOLO OBB model weights.
    confidence_treshold: float = 0.3
        Confidence threshold for detection.
    image_size: int = 1024
        YOLO model input size.
    device: str = "cpu"
        Which device to run on: e.g. 'cpu' or 'cuda:0'
    slice_height: Optional[int] = None
        Height of SAHI slices, defaults to image_size.
    slice_width: Optional[int] = None
        Width of SAHI slices, defaults to image_size.
    overlap_height_ratio: float = 0.1
        Overlap of SAHI slices in height.
    overlap_width_ratio: float = 0.1
        Overlap of SAHI slices in width.
    classes_to_keep: Optional[List[int]] = None
        Which target classes to keep. Defaults to all.
    class_agnostic: bool = False
        Whether to be class agnostic (e.g. when only caring about 'vehicles',
        classes for 'car' and 'truck' are treated as one).
    """

    def __init__(
        self,
        yolo_model_weights_path: str,
        confidence_treshold: float = 0.3,
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
            confidence_threshold=confidence_treshold,
            device=device,  # or 'cuda:0' if GPU is available
            image_size=image_size,
        )

        self.slice_height = slice_height or image_size
        self.slice_width = slice_width or image_size
        self.overlap_height_ratio = overlap_height_ratio
        self.overlap_width_ratio = overlap_width_ratio

        if classes_to_keep is not None:
            self.classes_to_exclude = [
                class_id
                for class_id in OBB_CLASSES.keys()
                if class_id not in classes_to_keep
            ]
        else:
            self.classes_to_exclude = None

        self.class_agnostic = class_agnostic

    def predict(self, image: npt.NDArray) -> PredictionResult:
        """
        Run sliced prediction on one image. Results will be returned and also
        stored until the next call of `predict(...)`.

        Parameters
        ----------
        image: npt.NDArray
            Input image.

        Returns
        -------
        An object of type sahi.predict.:class:`~sahi.predict.PredictionResult`.
        """
        self.last_result = get_sliced_prediction(
            image,
            self.detection_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
            exclude_classes_by_id=self.classes_to_exclude,
            postprocess_class_agnostic=self.class_agnostic,
            verbose=0,
        )
        return self.last_result

    def get_prediction_data(
        self,
        result: Optional[PredictionResult] = None,
    ) -> Dict[str, List]:
        """
        Converts the last (or given) :class:`~sahi.predict.PredictionResult` to
        a format compatible with YOLO OBB.

        Parameters
        ----------
        result: Optional[PredictionResult] = None
            The :class:`~sahi.predict.PredictionResult` to convert. Defaults to
            the results of the last call to `predict(...)`.

        Returns
        -------
        A dict with converted results:

        ```
        {
            "object_class": List[int] of predicted object classes,
            "confidence": List[int] of prediction confidence scores,
            "bounding_box": List[List] of prediction bounding boxes coordinates
        }
        ```
        """
        result = result or self.last_result

        obb_cls = [int(pred.category.id) for pred in result.object_prediction_list]
        obb_conf = [pred.score.value for pred in result.object_prediction_list]
        obb_boxes = np.concatenate(
            [
                [
                    np.reshape(pred.mask.segmentation, [4, 2])
                    for pred in result.object_prediction_list
                ]
            ]
        ).tolist()

        data = {
            "object_class": obb_cls,
            "confidence": obb_conf,
            "bounding_box": obb_boxes,
        }

        return data

    def get_number_of_slices_for_image(self, image: npt.NDArray) -> int:
        """
        Get the number of slices needed for a given image based on this
        SAHIModel's configuration.

        Parameters
        ----------
        image: npt.NDArray
            The input image.

        Returns
        -------
        The number of slices (int) that will be used for inference on this image.
        """
        slice_boxes = get_slice_bboxes(
            image_height=image.shape[0],
            image_width=image.shape[1],
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_height_ratio,
            overlap_width_ratio=self.overlap_width_ratio,
        )
        return len(slice_boxes)
