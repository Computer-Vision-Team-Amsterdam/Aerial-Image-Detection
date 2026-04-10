import os
import shutil
import tempfile
import time
from typing import List, Optional

import geopandas as gpd
import requests
from shapely.geometry.base import BaseGeometry

from aerial_image_detection import get_logger

logger = get_logger("pdok_downloader")

SLEEP_TIME = 5
REQUEST_TIMEOUT = 30


class PDOKDownloader:

    POST_URL = "https://api.pdok.nl/lv/bgt/download/v1_0/full/custom"
    POST_HEADERS = {"accept": "application/json", "Content-Type": "application/json"}

    GET_BASE_URL = "https://api.pdok.nl/lv/bgt/download/v1_0/full/custom/{}/status"
    GET_HEADERS = {
        "accept": "application/json",
    }

    DOWNLOAD_BASE_URL = "https://api.pdok.nl{}"

    def download_features_for_area(
        self,
        features: List[str],
        area: BaseGeometry,
        download_dir: str,
        extract_bgt_functions: Optional[dict[str, List[str]]] = None,
        suffix: Optional[str] = None,
        extension: Optional[str] = None,
    ) -> dict[str, str]:

        if len(features) > 1 and extract_bgt_functions is not None:
            raise AttributeError(
                "Extracting bgt_functions from multiple features is currently not supported."
            )

        post_data = {
            "featuretypes": features,
            "format": "citygml",
            "geofilter": area.wkt,
        }

        logger.info("Requesting download...")
        post_response = requests.post(
            url=self.POST_URL,
            headers=self.POST_HEADERS,
            json=post_data,
            timeout=REQUEST_TIMEOUT,
        )
        post_response.raise_for_status()

        download_request_id = post_response.json()["downloadRequestId"]

        get_headers = {
            "accept": "application/json",
        }

        download_ready = False
        download_location = None

        while not download_ready:
            get_response = requests.get(
                url=self.GET_BASE_URL.format(download_request_id),
                headers=get_headers,
                timeout=REQUEST_TIMEOUT,
            )
            get_response.raise_for_status()
            logger.debug(
                f"Status: {get_response.json()['status']} ({get_response.json()['progress']}%)"
            )

            if get_response.json()["status"] == "COMPLETED":
                download_location = get_response.json()["_links"]["download"]["href"]
                download_ready = True
            else:
                time.sleep(SLEEP_TIME)

        download_url = self.DOWNLOAD_BASE_URL.format(download_location)

        os.makedirs(download_dir, exist_ok=True)

        logger.info(f"Downloading and extracting {download_url}...")
        extracted_files = self._download_and_unzip_gml(
            url=download_url, download_dir=download_dir, suffix=suffix
        )

        output_files: dict[str, str] = dict()

        if extract_bgt_functions is not None:
            logger.info("Cropping results to area and extracting bgt functions...")
        else:
            logger.info("Cropping results to area...")
        for file in extracted_files:
            cropped_files = self._crop_and_extract(
                file_path=os.path.join(download_dir, file),
                target_shape=area,
                extension=extension,
                extract_bgt_functions=extract_bgt_functions,
            )
            output_files.update(cropped_files)

        return output_files

    @classmethod
    def _crop_and_extract(
        cls,
        file_path: str,
        target_shape: BaseGeometry,
        extension: Optional[str] = None,
        extract_bgt_functions: Optional[dict[str, List[str]]] = None,
    ) -> dict[str, str]:
        gdf = gpd.read_file(file_path)
        gdf_cropped = gdf[gdf.intersects(target_shape)]

        if extension is not None:
            os.remove(file_path)
            path, _ = os.path.splitext(file_path)
            if os.path.exists(f"{path}.gfs"):
                os.remove(f"{path}.gfs")
            file_path = f"{path}{extension}"

        gdf_cropped.to_file(filename=file_path)

        if extract_bgt_functions is not None:
            output_files = cls._extract_bgt_functions(file_path, extract_bgt_functions)
        else:
            output_files = {"all": file_path}

        return output_files

    @classmethod
    def _extract_bgt_functions(
        cls, file_path: str, extract_bgt_functions: dict[str, List[str]]
    ) -> dict[str, str]:
        extracted_files: dict[str, str] = dict()
        gdf = gpd.read_file(file_path)
        for group_name, functions in extract_bgt_functions.items():
            logger.debug(f"Extracting functions {functions} to {group_name}...")
            group_gdf = gdf[gdf["function"].isin(functions)]
            logger.debug(f"Found {len(group_gdf)} items.")

            path, ext = os.path.splitext(file_path)
            group_file_path = f"{path}_{group_name}{ext}"
            group_gdf.to_file(filename=group_file_path)
            logger.debug(f"Result saved in {group_file_path}")

            extracted_files[group_name] = group_file_path

        os.remove(file_path)

        return extracted_files

    @classmethod
    def _download_file(cls, url: str, file_path: str) -> None:
        logger.debug(f"Downloading {url} to {file_path}...")
        with requests.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
            with open(file_path, "wb") as f:
                shutil.copyfileobj(r.raw, f)

    @classmethod
    def _download_and_unzip_gml(
        cls, url: str, download_dir: str, suffix: Optional[str] = None
    ) -> List[str]:
        extracted_files = []
        with tempfile.TemporaryDirectory() as tmpdirname:
            with tempfile.NamedTemporaryFile(
                delete_on_close=False, suffix=".zip"
            ) as fp:
                cls._download_file(url=url, file_path=fp.name)
                fp.close()
                logger.debug(f"Unpacking archive to {download_dir}...")
                shutil.unpack_archive(fp.name, tmpdirname)
            file_list = os.listdir(tmpdirname)
            for file in file_list:
                if suffix is not None:
                    name, ext = os.path.splitext(file)
                    out_file_name = f"{name}_{suffix}{ext}"
                else:
                    out_file_name = os.path.basename(file)
                shutil.copy2(
                    src=os.path.join(tmpdirname, file),
                    dst=os.path.join(download_dir, out_file_name),
                )
                extracted_files.append(out_file_name)
                logger.debug(f"Extracted {file} to {out_file_name}")
        return extracted_files
