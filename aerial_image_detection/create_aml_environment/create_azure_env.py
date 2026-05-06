from aml_interface.aml_interface import AMLInterface

from aerial_image_detection import settings


def main():
    """
    This scripts creates an AML environment.
    """
    AMLInterface().create_aml_environment(
        env_name=settings["aml_experiment_details"]["env_name"],
        build_context_path="aerial_image_detection/create_aml_environment",
        dockerfile_path="Dockerfile",
        build_context_files=["pyproject.toml"],
    )


if __name__ == "__main__":
    main()
