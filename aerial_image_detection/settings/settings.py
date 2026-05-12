from cvtoolkit.settings.settings_helper import GenericSettings, Settings
from pydantic import BaseModel

from aerial_image_detection.settings.settings_schema import (
    AerialImageDetectionSettingsSpec,
)


class AerialImageDetectionSettings(Settings):  # type: ignore
    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = AerialImageDetectionSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)
