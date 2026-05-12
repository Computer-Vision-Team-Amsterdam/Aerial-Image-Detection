import json

import geopandas as gpd
import requests

from aerial_image_detection import logger
from aerial_image_detection.constants import RD_CRS, WGS84_CRS


class CityAreaHandler:
    """
    Helper class that queries the data.amsterdam.nl API for city area geometries
    and processes the results.

    Usage
    -----

    ```
        amsterdam_buurten_gdf = CityAreaHandler().get_city_area_gdf()
    ```
    """

    def __init__(self) -> None:
        self._query_and_process_stadsdelen()

    def _query_and_process_stadsdelen(self):
        """
        Query the API for 'buurten', 'wijken', 'stadsdelen' within Amsterdam and
        join them into one GeoDataFrame.
        """

        buurten = self.query_gebieden_api(scale="buurten")
        wijken = self.query_gebieden_api(scale="wijken")
        stadsdelen = self.query_gebieden_api(scale="stadsdelen")

        self.city_area_gdf = (
            buurten[["geometry", "identificatie", "naam", "code", "ligtInWijkId"]]
            .rename(columns={"naam": "buurt_naam", "code": "buurt_code"})
            .join(
                other=wijken[["identificatie", "naam", "code", "ligtInStadsdeelId"]]
                .set_index("identificatie")
                .rename(columns={"naam": "wijk_naam", "code": "wijk_code"}),
                how="left",
                on="ligtInWijkId",
            )
            .join(
                other=stadsdelen[["identificatie", "naam", "code"]]
                .set_index("identificatie")
                .rename(columns={"naam": "stadsdeel_naam", "code": "stadsdeel_code"}),
                how="left",
                on="ligtInStadsdeelId",
            )
        )
        self.city_area_gdf["gemeente_naam"] = "Amsterdam"

    @classmethod
    def query_gebieden_api(cls, scale: str) -> gpd.GeoDataFrame:
        """
        Query the API for city area geometries of a given scale and process the
        results.

        Parameters
        ----------
        scale: str
            One of: 'buurten', 'wijken', 'stadsdelen'.

        Returns
        -------
        A GeoDataFRame with the city areas.
        """
        if not (scale in ("buurten", "wijken", "stadsdelen")):
            raise ValueError(
                "Argument `scale` expected to be one of `['buurten', 'wijken', 'stadsdelen']`, "
                f"found `{scale}` instead."
            )

        url = f"https://api.data.amsterdam.nl/v1/gebieden/{scale}?_format=geojson"
        logger.info(f"Querying gebieden API for {scale}...")
        try:
            result = requests.get(url, timeout=5)
            result.raise_for_status()

            city_area_gdf = gpd.GeoDataFrame.from_features(
                features=json.loads(result.content), crs=WGS84_CRS
            ).to_crs(RD_CRS)
            logger.debug(f"Query successful, {len(city_area_gdf)} {scale} returned.")
            return city_area_gdf
        except requests.exceptions.RequestException as e:
            logger.error(f"Error querying gebieden API: {e}")
            raise e

    def get_city_area_gdf(self) -> gpd.GeoDataFrame:
        """Returns the gathered city areas as GeoDataFrame."""
        return self.city_area_gdf
