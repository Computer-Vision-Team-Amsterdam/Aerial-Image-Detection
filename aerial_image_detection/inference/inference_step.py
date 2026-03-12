import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from aerial_image_detection import logger, settings  # noqa: E402
from aerial_image_detection.inference.aerial_image_inference import (  # noqa: E402
    AerialImageInference,
)

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="inference_step",
    display_name="Run inference using SAHI and YOLO model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def run_inference(
    inference_data_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Run inference using a pretrained YOLO model on a chosen set of images.

    Parameters
    ----------
    inference_data_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of images to run inference on. The optional sub-folder
        structure will be preserved in the output.
    model_weights_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of the model weights.
    output_dir: Output(type=AssetTypes.URI_FOLDER)
        Location where output will be stored. Depending on the config settings
        this can be annotation labels as .txt files, images with blurred
        sensitive classes and bounding boxes, or both.
    """
    logger.info("Starting inference step.")

    inference_settings = settings["inference"]
    logger.debug("Inference settings from config.yml:")
    logger.debug(inference_settings)

    inference_step = AerialImageInference(
        images_folder=inference_data_dir,
        output_folder=output_dir,
        model_path=os.path.join(
            model_weights_dir, inference_settings["inputs"]["model_name"]
        ),
        inference_settings=inference_settings,
    )

    inference_step.run()
