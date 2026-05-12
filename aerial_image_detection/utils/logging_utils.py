import logging
from typing import Any, Dict, List

from azure.monitor.opentelemetry import configure_azure_monitor


class LoggingConfigurer:
    """
    Configures logging settings according to the provided configuration yaml file.

    Attributes
    ----------
    logging_cfg : dict
        Logging configuration dictionary containing the following keys:
        - loglevel_own: Logging level to set for the specified packages.
        - own_packages: List of packages to configure logging for.
        - basic_config: Dictionary containing basic logging configuration settings:
            - level: Logging level (e.g., INFO, DEBUG).
            - format: Format string for log messages.
            - datefmt: Date format for log messages.
        - ai_instrumentation_key: Azure Application Insights instrumentation key.


    packages : list of str
        List of packages to configure logging for, including the optional additional package.

    instrumentation_key : str
        Azure Application Insights instrumentation key.

    console_handler : logging.StreamHandler
        Handler for outputting logs to the console.

    Example usage:
    config = {
        "logging": {
            "loglevel_own": "INFO",
            "own_packages": ["a", "b"],
            "basic_config": {
                "level": "WARNING",
                "format": "%(asctime)s|||%(levelname)-8s|%(name)s|%(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "ai_instrumentation_key": "INSERT_KEY_HERE"
        }
    }

    azureLoggingConfigurer = AzureLoggingConfigurer(config["logging"])
    azureLoggingConfigurer.setup_oor_logging()
    logger = logging.getLogger("a")
    logger.info("Message with logger a")
    """

    def __init__(self, logging_cfg: Dict[str, Any], pkg_name: str = None):
        """
        Initializes the AzureLoggingConfigurer instance with the provided logging configuration.

        Parameters
        ----------
        logging_cfg : dict
            Logging configuration settings.

        pkg_name : str, optional
            Additional package name to configure logging for.
        """
        self.logging_cfg = logging_cfg
        logging.basicConfig(**self.logging_cfg["basic_config"])
        self.packages = self.logging_cfg["own_packages"] + (
            [pkg_name] if pkg_name else []
        )

        self.instrumentation_key = self.logging_cfg["ai_instrumentation_key"]
        if self.instrumentation_key is not None:
            configure_azure_monitor(
                connection_string=self.instrumentation_key,
            )

    def _setup_basic_logging(self, additional_handlers: List[logging.Handler] = []):
        """
        Sets up basic logging configurations for the specified packages.

        This method sets up logging configurations for the specified packages, including the addition of
        AzureLogHandler for sending logs to Azure Application Insights, and optional additional handlers.

        Parameters
        ----------
        additional_handlers : list of logging.Handler, optional
                              Additional logging handlers to be added to the logger.
        """
        for pkg in self.packages:
            pkg_logger = logging.getLogger(pkg)
            pkg_logger.setLevel(self.logging_cfg["loglevel_own"])

            if len(pkg_logger.handlers) != len(additional_handlers) + 1:
                for handler in additional_handlers:
                    pkg_logger.addHandler(handler)

    def setup_logging(self):
        """
        Set up logging configurations.
        """
        if self.instrumentation_key is not None:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.logging_cfg["loglevel_own"])
            self._setup_basic_logging(additional_handlers=[console_handler])
        else:
            self._setup_basic_logging()
