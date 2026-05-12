import os

from aml_interface.aml_interface import AMLInterface
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from aerial_image_detection import settings
from aerial_image_detection.inference.inference_step import run_inference

aml_interface = AMLInterface()


@pipeline()
def inference_pipeline():
    """AzureML pipeline to run aerial image detection."""

    input_datastore_path = aml_interface.get_datastore_full_path(
        settings["inference"]["input_folder"]["datastore"]
    )
    project_datastore_path = aml_interface.get_datastore_full_path(
        settings["inference"]["project_folder"]["datastore"]
    )

    inference_data_rel_path = settings["inference"]["input_folder"][
        "inference_data_rel_path"
    ]
    model_weights_rel_path = settings["inference"]["project_folder"][
        "model_weights_rel_path"
    ]
    output_rel_path = settings["inference"]["project_folder"]["output_rel_path"]

    inference_data_path = os.path.join(input_datastore_path, inference_data_rel_path)
    inference_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=inference_data_path,
    )

    model_weights_path = os.path.join(project_datastore_path, model_weights_rel_path)
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=model_weights_path,
    )
    run_inference_step = run_inference(
        inference_data_dir=inference_data, model_weights_dir=model_weights
    )

    output_path = os.path.join(project_datastore_path, output_rel_path)
    run_inference_step.outputs.output_dir = Output(
        type="uri_folder", mode="rw_mount", path=output_path
    )

    return {}


def main() -> None:
    """
    Script to submit the aerial image inference pipeline to AzureML.
    """
    aml_interface.submit_pipeline_experiment(
        pipeline_function=inference_pipeline,
        experiment_name=settings["aml_experiment_details"]["experiment_name"],
        default_compute=settings["aml_experiment_details"]["compute_name"],
        show_log=False,
    )


if __name__ == "__main__":
    main()
