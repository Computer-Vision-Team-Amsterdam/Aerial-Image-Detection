import logging
import os

from aerial_image_detection.settings import AerialImageDetectionSettings
from aerial_image_detection.utils.logging_utils import LoggingConfigurer

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)
try:
    AerialImageDetectionSettings.set_from_yaml(config_path)
    settings = AerialImageDetectionSettings.get_settings()
except FileNotFoundError:
    logging.getLogger().warning(
        f"Config file `{config_path}` for AerialImageDetection not found."
    )

logging_configurer = LoggingConfigurer(settings["logging"])
logging_configurer.setup_logging()

logger = logging.getLogger("aerial_image_detection")
