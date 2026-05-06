import logging
import os
from typing import Optional

from aerial_image_detection.settings import AerialImageDetectionSettings
from aerial_image_detection.utils.logging_utils import LoggingConfigurer

# Parse settings from config.yml
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

# Set-up logging
logging_configurer = LoggingConfigurer(settings["logging"])
logging_configurer.setup_logging()
logger = logging.getLogger("aerial_image_detection")


def get_logger(child_name: Optional[str] = None) -> logging.Logger:
    """
    Get the base logger or a child class of the base logger.

    Parameters
    ----------
    child_name: Optional[str] = None
        Name of the child logger.
    """
    if child_name is not None:
        return logger.getChild(child_name)
    else:
        return logger
