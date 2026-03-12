from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    compute_name: str = None
    env_name: str = None
    env_version: int = None
    src_dir: str = None


class LoggingSpec(SettingsSpecModel):
    loglevel_own: str = "INFO"
    own_packages: List[str] = [
        "__main__",
    ]
    extra_loglevels: Dict[str, str] = {}
    basic_config: Dict[str, Any] = {
        "level": "WARNING",
        "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    ai_instrumentation_key: Optional[str] = None


class InferenceModelParameters(SettingsSpecModel):
    model_name: str
    device: str = "cuda:0"
    img_size: Union[Tuple[int, int], int] = 1024
    conf: float = 0.1


class InferenceSAHIParameters(SettingsSpecModel):
    model_type: str = "ultralytics"
    slice_height: int = 1024
    slice_width: int = 1024
    overlap_height_ratio: float = 0.1
    overlap_width_ratio: float = 0.1
    postprocess_class_agnostic: bool = True


class InferenceSpec(SettingsSpecModel):
    input_folder: Dict[str, str]
    project_folder: Dict[str, str]
    model_params: InferenceModelParameters
    sahi_params: InferenceSAHIParameters
    target_area: Dict[str, str] = {"gemeente": "Amsterdam"}
    target_classes: List[int] = None
    target_classes_conf: Optional[float] = None


class AerialImageDetectionSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    logging: LoggingSpec = LoggingSpec()
    inference: InferenceSpec
