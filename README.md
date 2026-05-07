# Aerial-Image-Detection

This repository can be used to run object detection models on aerial imagery in the form of GeoTIFF files. It consists of two parts:
1. **Object detection**: use a [SAHI](https://github.com/obss/sahi) compatible model such as [YOLO OBB
    model](https://docs.ultralytics.com/tasks/obb/) to infer the location of objects on aerial images.
2. **Post-processing and analysis**: filter the detections to decrease FPs, and enrich the data with geospatial sources.

The object detection code can be run standalone (see [notebook](notebooks/aerial_image_pipeline.ipynb)), or as AzureML pipeline. The postprocessing step runs in a [notebook](notebooks/detection_postprocessing.ipynb).


## Context

This code was developed with the goal to count vehicles, in particular those that are located in pedestrian zones for which a permit is required. This analysis can, for example, be used to inform decision making related to permit policies. Knowing how many vehicles are located in pedestrian zones at any one time may tell something about the available space left for new permits, and comparing such data across different neighborhoods can similarly be informative.

As such, the code is designed to work with Dutch public data sources, in particular:
- Aerial imagery which can be downloaded from https://www.beeldmateriaal.nl/
- [BGT](https://www.kadaster.nl/zakelijk/registraties/basisregistraties/bgt) topographical data (in particular parking bays and pedestrian zones). A [notebook](notebooks/download_bgt_data.ipynb) is provided to facilitate downloading the required data.

Some parts of the code are specific to the City of Amsterdam, in particular the [CityAreaHandler](aerial_image_detection/utils/city_area_handler.py). For use in a different city, this code should be adapted.


## Installation

### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/Aerial-Image-Detection.git
```

### 2. Install UV
We use UV as package manager, which can be installed using any method mentioned on [the UV webpage](https://docs.astral.sh/uv/getting-started/installation/).

The easiest option is to use their installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It is also possible to use pip:
```bash
pipx install uv
```

Afterwards, uv can be updated using `uv self update`.

### 3. Install dependencies
In the terminal, navigate to the project root (the folder containing `pyproject.toml`), then use UV to create a new virtual environment and install the dependencies.

```bash
# Create the environment locally in the folder .venv
uv venv --python 3.12

# Activate the environment
source .venv/bin/activate 

# Install dependencies
uv pip install -r pyproject.toml --extra dev
```


## Usage

All configuration is done through the [config.yml](condig.yml) file. Adapt this to your needs.

### 1. Run the detection pipeline

The easiest option is to use the [notebook](notebooks/aerial_image_pipeline.ipynb) provided to run the code locally. If you have a GPU available, this will work fine. On a CPU, it might be slow.

Another option is to use AzureML and run the code as a pipeline. First, create a docker container that can be used to run the code. This will build the provided [Dockerfile](aerial_image_detection/create_aml_environment/Dockerfile).

```bash
uv run python aerial_image_detection/create_aml_environment/create_azure_env.py
```

Then, submit the pipeline:

```bash
uv run python aerial_image_detection/submit_inference_pipeline.py
```

### 2. Run the post-processing code

This code runs locally in a [notebook](notebooks/detection_postprocessing.ipynb). It has been optimized for performance on a standard laptop, e.g. analysis for the whole of Amsterdam runs in roughly 5-10 minutes.


## Contributing

If you wish to contribute any updates, improvements, or extensions, feel free to make a pull request.

We use pre-commit hooks to to ensure that all committed code is valid and consistently formatted. We use UV to manage pre-commit as well.

```bash
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Install pre-commit hooks
pre-commit install

# Optional: update pre-commit hooks
pre-commit autoupdate

# Run pre-commit hooks using
bash .git/hooks/pre-commit
```
