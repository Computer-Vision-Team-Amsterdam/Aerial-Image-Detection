import json

import geopandas as gpd
import requests


class CityAreaHandler:
    """
    Query the API for city area geometries and process the results.
    """

    RD_CRS = "EPSG:28992"
    WGS84_CRS = "EPSG:4326"

    def __init__(self) -> None:
        self._query_and_process_stadsdelen()

    def _query_and_process_stadsdelen(self):
        """
        Query the API for city area geometries and process the results.

        The function runs a query to fetch geometries for city areas and
        converts them to a GeoPandas dataframe.
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
        if not (scale in ("buurten", "wijken", "stadsdelen")):
            raise ValueError(
                "Argument `scale` expected to be one of `['buurten', 'wijken', 'stadsdelen']`, "
                f"found `{scale}` instead."
            )

        url = f"https://api.data.amsterdam.nl/v1/gebieden/{scale}?_format=geojson"
        print(f"Querying gebieden API for {scale}...")
        try:
            result = requests.get(url, timeout=5)
            result.raise_for_status()

            city_area_gdf = gpd.GeoDataFrame.from_features(
                features=json.loads(result.content), crs=cls.WGS84_CRS
            ).to_crs(cls.RD_CRS)
            print(f"Query successful, {len(city_area_gdf)} {scale} returned.")
            return city_area_gdf
        except requests.exceptions.RequestException as e:
            print(f"Error querying gebieden API: {e}")
            raise e

    def get_city_area_gdf(self) -> gpd.GeoDataFrame:
        return self.city_area_gdf
