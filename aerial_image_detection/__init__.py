import logging
import os

from aml_interface.aml_interface import AMLInterface

from aerial_image_detection.settings import AerialImageDetectionSettings

logger = logging.getLogger("inference_pipeline")

aml_interface = AMLInterface()

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)
try:
    AerialImageDetectionSettings.set_from_yaml(config_path)
    settings = AerialImageDetectionSettings.get_settings()
except FileNotFoundError:
    logger.warning(f"Config file `{config_path}` for AerialImageDetection not found.")
