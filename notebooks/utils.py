"""
Utilities used by example notebooks
"""

from __future__ import annotations

from typing import Any

from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import osmnx as ox
import os
import geopandas as gpd
from shapely.geometry import Point, shape, Polygon, MultiPolygon
from pyproj import CRS, Transformer
import json
import rasterio.features

from sentinelhub import (
    SHConfig,
    CRS as sentinelCRS,
    BBox,
    DataCollection,
    Geometry,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
)


def plot_image(
    image: np.ndarray,
    factor: float = 1,
    clip_range: tuple[float, float] | None = None,
    **kwargs: Any,
) -> None:
    """Utility function for plotting RGB images."""
    _, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        ax.imshow(np.clip(image * factor, *clip_range), **kwargs)
    else:
        ax.imshow(image * factor, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])


def get_india_states():
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    aoi_dir = os.path.join(parent_dir, "aoi")

    india_path = os.path.join(aoi_dir, "india1.json")
    india_states = gpd.read_file(india_path)
    return india_states

def get_data_directory():
    return r"D:\data"

def get_lakes(
    longitude: float, latitude: float, distance: int, area: int = 1000000
) -> gpd.GeoDataFrame:

    tags = {"natural": "water"}  # Searching for natural water bodies
    ox.settings.use_cache = False
    waterbodies = ox.features_from_point(
        (latitude, longitude), tags=tags, dist=distance
    )
    allowed_values = ["pond", "reservoir", "lake", "basin"]

    lakes = waterbodies[waterbodies["water"].isin(allowed_values)]
    lakes = lakes[["name", "geometry", "water"]]
    
    # Ensure geometries are valid and not empty
    lakes = lakes[lakes["geometry"].notna()]
    lakes = lakes[lakes.geometry.is_valid]

    # Add centroid coordinates
    lakes["geometry_ws"] = lakes["geometry"]

    # Determine the appropriate UTM zone and reproject
    zone_number = int((longitude + 180) / 6) + 1
    utm_crs = CRS(f"EPSG:326{zone_number}")
    ws_crs = "EPSG:4326"
    lakes = lakes.to_crs(utm_crs)

    transformer_to_utm = Transformer.from_crs(ws_crs, utm_crs, always_xy=True)
    reference_point_utm = transformer_to_utm.transform(longitude, latitude)

    lakes["area"] = lakes.geometry.area
    lakes["centroid"] = lakes.geometry.centroid
    lakes["distance"] = lakes["centroid"].apply(
        calculate_distance, point=reference_point_utm
    )
    lakes["centroid_ws"] = lakes["centroid"].to_crs(epsg=4326)

    lakes = lakes[lakes["area"] > area]

    lakes = lakes.reset_index()

    new_rows = []
    for i, row in lakes.iterrows():
        geom_ws = row["geometry_ws"]
        geom_crs = row["geometry"]
        if isinstance(geom_ws, MultiPolygon) & isinstance(geom_crs, MultiPolygon):
            if len(geom_ws.geoms) == len(geom_crs.geoms):
                transformer_to_ws = Transformer.from_crs(utm_crs, ws_crs)
                for polygon_index in range(len(geom_ws.geoms)):
                    new_row = {}
                    new_row["osmid"] = f"{row["osmid"]}_{polygon_index}"
                    new_row["geometry_ws"] = geom_ws.geoms[polygon_index]
                    new_row["geometry"] = geom_crs.geoms[polygon_index]
                    new_row["area"] = new_row["geometry"].area
                    new_row["centroid"] = new_row["geometry"].centroid
                    new_row["distance"] = calculate_distance(new_row["centroid"], reference_point_utm)
                    longitude, latitude = transformer_to_ws.transform(new_row["centroid"].x,new_row["centroid"].y)
                    new_row["centroid_ws"] = Point(latitude, longitude)
                    new_rows.append(new_row)

            lakes.drop(i, inplace = True)

        elif isinstance(geom_ws, Polygon) & isinstance(geom_crs, Polygon):
            pass
        else:
            lakes.drop(i, inplace = True)
    if new_rows:
        new_rows_gdf = gpd.GeoDataFrame(new_rows, geometry='geometry', crs=utm_crs)
        lakes = gpd.GeoDataFrame(pd.concat([lakes, new_rows_gdf], ignore_index=True), crs=utm_crs)
        lakes = lakes.sort_values(by="area", ascending=False)
        lakes = lakes[lakes["area"] > area]

    for i, row in lakes.iterrows():
        polygon = row["geometry_ws"]
        osmid = row["osmid"]
        coords = list(polygon.exterior.coords)
        geojson = {"type": "Polygon", "coordinates": [coords]}
        geojson_filename = f"../aoi/{osmid}.geojson"
        with open(geojson_filename, "w") as f:
            json.dump(geojson, f, indent=4)

    lakes = lakes[
        [
            "osmid",
            "name",
            "geometry_ws",
            "area",
            "distance",
            "centroid_ws",
            "water",
        ]
    ]
    return lakes


def get_all_lakes(df, distance, area=1000000, drop_duplicated=True):
    dataframes = []
    ox.settings.use_cache = False
    for i, row in df.iterrows():
        print(f"Finding lakes of university number {i}")
        longitude = row["longitude"]
        latitude = row["latitude"]
        university_lakes = get_lakes(longitude, latitude, distance, area)
        if len(university_lakes) != 0:
            dataframes.append(university_lakes)
            university_lakes["university"] = row["name"]

    all_lakes = pd.concat(dataframes, ignore_index=True)
    all_lakes = all_lakes.sort_values(by="area", ascending=False)
    plot_indian_lakes(df, all_lakes)
    if drop_duplicated == True:
        all_lakes = drop_duplicated_lakes(all_lakes)

    all_lakes["centroid_long"] = all_lakes["centroid_ws"].apply(lambda lake: lake.x)
    all_lakes["centroid_lat"] = all_lakes["centroid_ws"].apply(lambda lake: lake.y)
    all_lakes = all_lakes.drop(columns=["geometry_ws", "centroid_ws"])
    csv_filename = os.path.join(get_data_directory(), "all_lakes.csv")
    all_lakes[["pixel_size", "pixel_count"]] = all_lakes["osmid"].apply(lambda osmid: pd.Series(calculate_pixel_size_and_pixel_count(osmid)))
    all_lakes["total_pixel_area"] = all_lakes['pixel_count'] * (all_lakes["pixel_size"]**2)
    all_lakes.to_csv(csv_filename, index=False)

    return all_lakes


def calculate_distance(centroid_utm, point):
    return centroid_utm.distance(Point(point))


def calculate_crs(point: Point) -> int:
    longitude = point.coords[0][1]
    zone_number = int((longitude + 180) / 6) + 1
    utm_crs = int(f"326{zone_number}")
    return utm_crs


def transform_geometries_by_crs(lakes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # Crear una lista para almacenar las geometrías transformadas
    transformed_geometries = []

    for idx, row in lakes.iterrows():
        # Obtener el CRS para el registro actual
        crs_code = row["calculated_crs"]

        try:
            # Crear un CRS a partir del código EPSG
            crs = CRS(f"EPSG:{int(crs_code)}")

            # Crear un GeoDataFrame temporal con el CRS actual
            temp_gdf = gpd.GeoDataFrame(
                {"geometry": [row["geometry"]]}, crs="EPSG:4326"
            )
            temp_gdf = temp_gdf.to_crs(crs)

            # Obtener la geometría transformada
            transformed_geometry = temp_gdf.iloc[0].geometry
        except Exception as e:
            print(f"Error al transformar la geometría en el índice {idx}: {e}")
            transformed_geometry = row[
                "geometry"
            ]  # Dejar la geometría sin transformar en caso de error

        # Agregar la geometría transformada a la lista
        transformed_geometries.append(transformed_geometry)

    lakes_transformed = lakes.copy()
    lakes_transformed["geometry"] = transformed_geometries

    return lakes_transformed


def drop_duplicated_lakes(gdf):
    idx = gdf.groupby("osmid")["distance"].idxmin()
    filtered_lakes = gdf.loc[idx]

    # Resetear el índice del DataFrame resultante
    filtered_lakes = filtered_lakes.reset_index(drop=True)
    return filtered_lakes


def plot_indian_lakes(universities, lakes):
    india_states = get_india_states()
    universities["geometry"] = universities.apply(
        lambda row: Point(row["longitude"], row["latitude"]), axis=1
    )
    universities_gdp = gpd.GeoDataFrame(universities, geometry="geometry")
    lakes_gdp = gpd.GeoDataFrame(lakes["centroid_ws"], geometry = "centroid_ws")
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Plotear el mapa base de la India
    india_states.plot(ax=ax, color="lightgrey", edgecolor="black")
    lakes_gdp.plot(ax=ax, color="blue", markersize=20, label="Lakes")

    universities_gdp.plot(ax=ax, color="red", markersize=20, label="Universities")
    plt.legend()

    # Añadir título
    plt.title("Lakes and Universities from India")

    # Mostrar el plot
    plt.show()


def get_config():
    config = SHConfig()
    #Berta
    # config.sh_client_id = "a2bf076a-71dc-49b7-9b5b-5aa18c6469b5"
    # config.sh_client_secret = "nGK7OacxYFmTSC517fCxFxsEPD6FTt31"
    #DavidMates1
    # config.sh_client_id = "d0d13e6b-b1cf-44f3-b9ac-5961eb728b5b"
    # config.sh_client_secret = "Y5ParWR5M95dbrHWuHjVrOzayXj7eJpS"
    #FernandoMates
    # config.sh_client_id = "7b09811c-f727-451f-b631-c5711bd1ff86"
    # config.sh_client_secret = "IIBnLjK0TAWFqCDZSQax0cVok4VBPNja"
    # #Roberto
    # config.sh_client_id = "463443bf-6ffc-4e9f-abb6-01b1e5a71e78"
    # config.sh_client_secret = "kNWd3ZLmzBHGujXIUUpidXk5K6oFIndk"
    #Alejandro
    # config.sh_client_id = "00ebb2cc-3c8c-46fa-8aff-d1cae79a422b"
    # config.sh_client_secret = "goPBBsJbeyPkiLndWqBUxLIjsbngqgWc"
    #Taila
    # config.sh_client_id = "d4f55ce5-a803-4ab5-b680-bc5f19957d15"
    # config.sh_client_secret = "2JOEOzOxfysriqhfCeIO0846I1e9ULp7"
    #Taila2
    config.sh_client_id = "4f4361f4-42eb-495b-b036-8c4a0469c4ba"
    config.sh_client_secret = "DpJgxZWWJwGLhnTUG9owqFN1tfbhKO6g"


    return config

def get_image_from_lake(osmid, type_image, date, resolution):

    config = get_config()
    date_string = date.strftime("%Y-%m-%d")
    if type_image.upper() == "WATER":
        type_image = "ndwi"
    elif type_image.upper() == "CHL":
        type_image = "chlorophyll"
    elif type_image.upper() == "TRUE":
        type_image = "true-color"
    elif type_image.upper() == "CLOUD":
        type_image = "cloud"
    elif type_image.upper() == "SIZE":
        type_image = "size"

    data_directory = get_data_directory()
    directory = f"{data_directory}/{osmid}"
    image_path = f"{directory}/{osmid}_{type_image.lower()}_{date_string}.npy"
    if os.path.exists(image_path):
        save = False
        image = np.load(image_path)
        return image
    else:
        save = True
        ## Get evalscript
        current_directory = os.getcwd()
        parent_directory = os.path.dirname(current_directory)
        evalscripts = os.path.join(parent_directory, "evalscripts")

        evalscript_path = os.path.join(evalscripts, f"{type_image}.js")
        with open(evalscript_path, "r") as file:
            evalscript = file.read()

        ## Get geojson
        aois_path = os.path.join(parent_directory, "aoi")
        geojson_path = os.path.join(aois_path, f"{osmid}.geojson")
        gdf = gpd.read_file(geojson_path)

        with open(geojson_path, "r") as file:
            aoi = json.load(file)

        geometry = Geometry(aoi, crs=sentinelCRS.WGS84)

        minx, miny, maxx, maxy = gdf.total_bounds
        bbox = BBox(bbox=[minx, miny, maxx, maxy], crs="EPSG:4326")
        polygon_size = bbox_to_dimensions(bbox, resolution=resolution)

        # request_color = SentinelHubRequest(
        #     evalscript=evalscript,
        #     input_data=[
        #         SentinelHubRequest.input_data(
        #             data_collection=DataCollection.SENTINEL2_L2A,
        #             time_interval=(date_string, date_string),
        #         )
        #     ],
        #     responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        #     bbox=bbox,
        #     geometry=geometry,
        #     size=polygon_size,
        #     config=config,
        # )

        request_params = {
            "evalscript": evalscript,
            "input_data": [
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(date_string, date_string),
                    mosaicking_order="mostRecent",
                    other_args={"dataMask": True},
                )
            ],
            "responses": [SentinelHubRequest.output_response("default", MimeType.PNG)],
            "bbox": bbox,
            "size": polygon_size,
            "config": config,
        }

        if type_image != "true-color":
            with open(geojson_path, "r") as file:
                aoi = json.load(file)
            geometry = Geometry(aoi, crs=sentinelCRS.WGS84)
            request_params["geometry"] = geometry

        request_color = SentinelHubRequest(**request_params)

        raw_image = request_color.get_data()
        image = raw_image[0]

        if save == True:
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save(
                f"{directory}/{osmid}_{type_image.lower()}_{date_string}.npy",
                image,
            )
        return image


def get_all_images_from_lake(osmid, type_image, from_date, to_date, resolution):
    images = []
    i = 0
    current_date = from_date
    while current_date < to_date:

        image = get_image_from_lake(osmid, "chl", current_date, resolution)
        if not np.all(image == 0):
            if type_image.lower() == "chl":
                images.append(image)
            else:
                image = get_image_from_lake(osmid, type_image, current_date, resolution)
            selected_day = current_date + timedelta(days=5)
            break
        i += 1
        current_date = from_date + timedelta(days=i)

    while selected_day <= to_date:
        image = get_image_from_lake(osmid, type_image, selected_day, resolution)
        images.append(image)
        selected_day += timedelta(days=5)

    return images

# def generate_historical_all_type_images_from_lake(
#     osmid, from_date, to_date, save=False
# ):
#     historical_images = []
#     i = 0
#     current_date = from_date
#     while current_date < to_date:
#         image = get_image_from_lake(osmid, "chl", current_date)
#         if not np.all(image == 0):
#             selected_day = current_date
#             break
#         i += 1
#         current_date = from_date + timedelta(days=i)

#     while selected_day <= to_date:
#         images = generate_all_type_images_from_lake(osmid, selected_day, save)
#         historical_images.extend(images)
#         selected_day += timedelta(days=5)

#     return historical_images


# def generate_all_type_images_from_lake(osmid, current_date):
#     images = []
#     type_images = ["TRUE", "WATER", "CLOUD", "CHL"]
#     for each_type in type_images:
#         image = get_image_from_lake(osmid, each_type, current_date)
#         images.append(image)
#     return images


## Transforma cada píxel de la imagen del chl en una categoría
def categorize_pixel(pixel, reference_colors):
    # Only consider the RGB part of the pixel
    distances = np.linalg.norm(reference_colors - pixel[:3], axis=1)
    return np.argmin(distances)


def analyze_chl_image(image):
    reference_colors = np.array(
        [
            [1, 4, 42],  # Very dark blue
            [0, 106, 78],  # Dark teal
            [124, 250, 0],  # Bright yellow-green
            [241, 215, 27],  # Mustard yellow
            [255, 0, 0],  # Pure red
        ]
    )

    white_pixels_mask = np.all(image[:, :, :3] == [255, 255, 255], axis=-1)
    black_pixels_mask = np.all(image[:, :, :3] == [0, 0, 0], axis=-1)

    non_black_or_white_mask = ~white_pixels_mask & ~black_pixels_mask

    non_black_or_white_pixels = image[non_black_or_white_mask]
    non_black_or_white_pixels = non_black_or_white_pixels.reshape(-1, image.shape[2])

    categorized_pixels = [
        categorize_pixel(pixel, reference_colors) for pixel in non_black_or_white_pixels
    ]
    category_counts = np.bincount(categorized_pixels, minlength=len(reference_colors))

    return category_counts


def analyze_water_image(image):
    blue_pixels_mask = (
        (image[:, :, 2] > image[:, :, 0])
        & (image[:, :, 2] > image[:, :, 1])
        & (image[:, :, 3] != 0)
    )
    blue_pixels_count = np.sum(blue_pixels_mask)
    return blue_pixels_count


def analyze_cloud_image(image):
    red_pixels_mask = (
        (image[:, :, 0] > image[:, :, 1])
        & (image[:, :, 0] > image[:, :, 2])
        & (image[:, :, 0] > 190)
    )
    red_pixels_count = np.sum(red_pixels_mask)
    return red_pixels_count


def analyze_true_image(image):
    white_pixels_mask = (
        (image[:, :, 0] > 250)
        & (image[:, :, 1] > 250)
        & (image[:, :, 2] > 250)
    )
    white_pixels_count = np.sum(white_pixels_mask)
    total_pixels_count = image.shape[0] * image.shape[1]
    threshold_white = 0.10
    if white_pixels_count/total_pixels_count > threshold_white:
        return False
    return True


def analyze_image(image, type_image):
    if type_image == "TRUE":
        return analyze_true_image(image)
    elif type_image == "WATER":
        return analyze_water_image(image)
    elif type_image == "CLOUD":
        return analyze_cloud_image(image)
    elif type_image == "CHL":
        return analyze_chl_image(image)
    ## Raise error
    else:
        return None

def calculate_area(pixel_count, resolution):
    return pixel_count * resolution ** 2

def calculate_log(osmid, date_str, total_area, resolution, images):
    exists_true_image = analyze_true_image(images[0])
    # log = {
    #     "osmid": osmid,
    #     "day": date_str,
    #     "exists_true_image": exists_true_image}
    # return log
    if len(images) == 1:
        log = {
            "osmid": osmid,
            "day": date_str,
            "exists_true_image": exists_true_image,
            "total_pixel_area": total_area,
            # "total_pixel_count": total_pixel_count,
            # "is_valid" : False
        }
        return log
    elif len(images) == 4:
        water_image = images[1]
        cloud_image = images[2]
        chl_image = images[3]

        water_pixels_mask = (
            (water_image[:, :, 2] > water_image[:, :, 0])
            & (water_image[:, :, 2] > water_image[:, :, 1])
            & (water_image[:, :, 3] != 0)
        )
        water_pixels_count = np.sum(water_pixels_mask)
        water_area = calculate_area(water_pixels_count, resolution)

        cloud_pixels_mask = (
            (cloud_image[:, :, 0] > cloud_image[:, :, 1])
            & (cloud_image[:, :, 0] > cloud_image[:, :, 2])
            & (cloud_image[:, :, 0] > 190)
        )
        cloud_pixels_count = np.sum(cloud_pixels_mask)
        cloud_area = calculate_area(cloud_pixels_count, resolution)

        water_no_cloud_mask = water_pixels_mask & ~cloud_pixels_mask

        water_no_cloud_count = np.sum(water_no_cloud_mask)
        water_no_cloud_area = calculate_area(water_no_cloud_count, resolution)

        water_no_cloud_region = np.zeros_like(chl_image)
        water_no_cloud_region[water_no_cloud_mask] = chl_image[water_no_cloud_mask]

        chl_concentrations = analyze_chl_image(water_no_cloud_region)
        chl_very_low = chl_concentrations[0]
        chl_low = chl_concentrations[1]
        chl_moderate = chl_concentrations[2]
        chl_high = chl_concentrations[3]
        chl_very_high = chl_concentrations[4]

        chl_very_low_area = calculate_area(chl_very_low, resolution)
        chl_low_area = calculate_area(chl_low, resolution)
        chl_moderate_area = calculate_area(chl_moderate, resolution)
        chl_high_area = calculate_area(chl_high, resolution)
        chl_very_high_area = calculate_area(chl_very_high, resolution)

        no_chl_area = water_no_cloud_area - (chl_very_low_area + chl_low_area + chl_moderate_area + chl_high_area + chl_very_high_area)
        

        log = {
            "osmid": osmid,
            "day": date_str,
            "exists_true_image": exists_true_image,
            "total_pixel_area" : total_area,
            # "total_pixel_count": total_pixel_count,
            "water_area": water_area,
            "cloud_area": cloud_area,
            "water_with_no_clouds_area": water_no_cloud_area,
            "no_chl_area" : no_chl_area,
            "chl_very_low_area":chl_very_low_area,
            "chl_low_area":chl_low_area,
            "chl_moderate_area":chl_moderate_area,
            "chl_high_area":chl_high_area,
            "chl_very_high_area":chl_very_high_area
            }
        
        return log
    else:
        log = {"osmid":osmid, "day":date_str}
        return log


def get_lake_log(lake_row, current_date):
    images = []
    log = {""}
    osmid = lake_row["osmid"]
    resolution = lake_row["pixel_size"]
    # total_pixel_count = lake_row["pixel_count"]
    total_area = lake_row["total_pixel_area"]
    date_str = current_date.strftime("%Y-%m-%d")
    true_image = get_image_from_lake(osmid, "true", current_date, resolution)
    images.append(true_image)
    exists_true_image = analyze_true_image(true_image)
    if  exists_true_image == False:
        
        log = calculate_log(osmid, date_str, total_area, resolution, images)
        return log#, images

    water_image = get_image_from_lake(osmid, "water", current_date, resolution)
    images.append(water_image)
    cloud_image = get_image_from_lake(osmid, "cloud", current_date, resolution)
    images.append(cloud_image)
    chl_image = get_image_from_lake(osmid, "chl", current_date, resolution)
    images.append(chl_image)

    log = calculate_log(osmid, date_str, total_area, resolution, images)
    return log

def check_log(log):
    total_pixel_count = log["total_pixel_count"]
    cloud_pixels_count = log["cloud_pixels"]
    water_with_no_clouds_pixels = log["water_with_no_clouds_pixels"]
    threshold_cloud=0.10
    threshold_water_no_cloud = 0.25
    if (cloud_pixels_count/total_pixel_count < threshold_cloud) & (water_with_no_clouds_pixels>threshold_water_no_cloud):
        return True
    else:
        return False


def get_historical_lake_log(lake_dict, from_date, to_date, skip_count=1):
    logs = []
    # all_images = {}
    current_date = from_date
    i = 0
    while current_date < to_date:
        log = get_lake_log(lake_dict, current_date)
        if log["exists_true_image"] == True:
            selected_day = current_date
            selected_day_str = selected_day.strftime("%Y-%m-%d")
            # all_images[selected_day_str] = images
            logs.append(log)
            break
        i += 1
        current_date = from_date + timedelta(days=i)

    selected_day += timedelta(days=5 * skip_count)
    
    while selected_day <= to_date:
        print(selected_day)
        selected_day_str = selected_day.strftime("%Y-%m-%d")
        log = get_lake_log(lake_dict, selected_day)
        # all_images[selected_day_str] = images
        logs.append(log)
        selected_day += timedelta(days=5 * skip_count)

    lake_logs = pd.DataFrame(logs)
    return lake_logs#, all_images

def get_historical_all_lakes_logs(all_lakes, from_date, to_date, skip_count=1):
    new_logs = []
    for i, row in all_lakes.iterrows():
        print(i)
        osmid = row["osmid"]
        lake_logs = get_historical_lake_log(row,  from_date, to_date, skip_count)
        new_logs.append(lake_logs)

    new_logs_df = pd.concat(new_logs, ignore_index=True)

    data_directory = get_data_directory()
    old_logs_df = pd.read_csv(f"{data_directory}/all_saved_logs.csv")
    all_logs_df = pd.concat([new_logs_df, old_logs_df], ignore_index=True)

    all_logs_df.to_csv(f"{data_directory}/all_saved_logs.csv", index = False)
    all_logs_df.to_csv("../data/all_saved_logs.csv", index = False)

    return new_logs_df

def get_all_saved_logs(lakes):
    all_logs = []
    for i, row in lakes.iterrows():
        osmid = str(row["osmid"])
        total_pixel_area = row["total_pixel_area"]
        resolution = row["pixel_size"]
        data_dir = get_data_directory()
        osmid_dir = os.path.join(data_dir, osmid)
        true_image_files = [image_file for image_file  in os.listdir(osmid_dir) if image_file.endswith(".npy") and "true" in image_file.lower()]
        print(i, ": ", len(true_image_files))
        j = 0
        for true_image_file in true_image_files:
            parts = true_image_file.split("_")
            # print(parts)
            true_image_path = os.path.join(osmid_dir, true_image_file)
            images = [np.load(true_image_path)]
            date_str = parts[2]
            water_image_path = os.path.join(osmid_dir, f"{osmid}_ndwi_{date_str}")
            # print(water_image_path, os.path.exists(water_image_path))
            cloud_image_path =  os.path.join(osmid_dir,f"{osmid}_cloud_{date_str}")
            # print(cloud_image_path, os.path.exists(cloud_image_path))
            chl_image_path =  os.path.join(osmid_dir,f"{osmid}_chlorophyll_{date_str}")
            # print(chl_image_path, os.path.exists(chl_image_path))
            if os.path.exists(water_image_path) & os.path.exists(cloud_image_path) & os.path.exists(chl_image_path):
                water_image = np.load(water_image_path)
                images.append(water_image)
                cloud_image = np.load(cloud_image_path)
                images.append(cloud_image)
                chl_image = np.load(chl_image_path)
                images.append(chl_image)
            date_str = date_str.replace(".npy", "")
            log = calculate_log(osmid, date_str, total_pixel_area, resolution, images)
            all_logs.append(log)
    
    all_saved_logs = pd.DataFrame(all_logs)
    data_directory = get_data_directory()
    all_saved_logs.to_csv(f"{data_directory}/top_lakes_logs.csv")
    all_saved_logs.to_csv(r"../data/top_lakes_logs.csv")
    return all_saved_logs







def plot_all_type_images(osmid, current_date, images):
    if len(images) != 4:
        raise ValueError("La lista de imágenes debe contener exactamente 4 elementos.")

    fig, axes = plt.subplots(
        2, 2, figsize=(8, 8)
    )  # Crea una cuadrícula de 2x2 para las imágenes
    titles = ["True Color", "Water Body", "Cloud", "Chl index"]
    fig.suptitle(f"Lake {osmid}: {current_date.strftime('%Y-%m-%d')}", fontsize=16)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i], cmap="gray")  # Muestra cada imagen en la cuadrícula
        ax.set_title(titles[i], fontsize=12)
        ax.axis("off")  # Oculta los ejes para una mejor visualización

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajusta el espaciado entre los gráficos
    plt.show()


def plot_log(log, images):
    """
    Función para graficar 4 imágenes en un grid 2x2 y una tabla que ocupa toda la línea inferior.

    Parámetros:
    osmid (int): Identificador del lago.
    current_date (datetime): Fecha de la imagen.
    images (list): Una lista o array que contiene 4 imágenes.
    log (dict): Diccionario con los valores del registro del lago.
    """
    if len(images) == 4:

        # Crear una cuadrícula con 3 filas y 2 columnas, donde la última fila ocupará toda la fila para la tabla
        fig = plt.figure(constrained_layout=True, figsize=(10, 15))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[2, 2, 1])

        # Título general

        osmid = log["osmid"]
        current_date = log["day"]
        fig.suptitle(f"Lake {osmid}: {current_date}", fontsize=16)

        # Añadir imágenes a la cuadrícula
        axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]
        titles = ["True Color", "Water Body", "Cloud", "Chl index"]

        for i, ax in enumerate(axes):
            ax.imshow(images[i], cmap="gray")  # Muestra cada imagen
            ax.set_title(titles[i], fontsize=12)
            ax.axis("off")  # Oculta los ejes para una mejor visualización

        # Crear la tabla ocupando ambas columnas en la fila inferior
        ax_table = fig.add_subplot(gs[2, :])  # Esto toma toda la fila inferior
        ax_table.axis("off")  # Ocultar el cuadro alrededor de la tabla

        table_data = []
        for key, value in log.items():
            table_data.append([key, value])

        # Dibujar la tabla ocupando las dos columnas de la fila inferior
        table = ax_table.table(
            cellText=table_data,
            colLabels=["Description", "Value"],
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)

        plt.tight_layout()

        plt.show()


def plot_historical_lake_logs(logs, images):
    for index, row in logs.iterrows():
        date_str = row["day"]
        day_images = images[date_str]
        plot_log(row, day_images)

# def plot_all_historical_lake_logs(all_logs, all_images):
#     for index, row in all_logs.iterrows():
#         osmid = row["osmid"]
#         date_str = row["day"]
#         images = all_images[osmid][date_str]
#         plot_log(row, images)

def plot_all_historical_lake_logs(all_logs):
    for index, row in all_logs.iterrows():
        osmid = row["osmid"]
        date_str = row["day"]
        day = datetime.strptime(date_str, '%Y-%m-%d')
        true_image = get_image_from_lake(osmid, "true", day, 5)
        water_image = get_image_from_lake(osmid, "water", day, 5)
        cloud_image = get_image_from_lake(osmid, "cloud", day, 5)
        chl_image = get_image_from_lake(osmid, "chl", day, 5)
        images = [true_image, water_image, cloud_image, chl_image]
        plot_log(row, images)


def calculate_pixel_size_and_pixel_count(osmid):
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    aois_path = os.path.join(parent_directory, "aoi")
    geojson_path = os.path.join(aois_path, f"{osmid}.geojson")
    gdf = gpd.read_file(geojson_path)

    with open(geojson_path, "r") as file:
        aoi = json.load(file)

    geometry = Geometry(aoi, crs=sentinelCRS.WGS84)
    minx, miny, maxx, maxy = gdf.total_bounds
    bbox = BBox(bbox=[minx, miny, maxx, maxy], crs="EPSG:4326")
    pixel_size = 5
    polygon_size = bbox_to_dimensions(bbox, resolution=pixel_size)

    max_size = max(polygon_size)
    aux = max_size
    i = 1
    while aux > 2500:
        i += 1
        aux = max_size
        aux = aux / i

    pixel_size = 5 * i

    mask = rasterio.features.geometry_mask(
        [shape(geometry.geometry)],
        out_shape=(polygon_size[1], polygon_size[0]),  # (alto, ancho)
        transform=rasterio.transform.from_bounds(
            minx, miny, maxx, maxy, polygon_size[0], polygon_size[1]
        ),
        invert=True,
    )

    pixel_count = int(np.sum(mask) / (i**2))

    return pixel_size, pixel_count

def get_transformed_image(images):
    true_image = images[0][:, :, :3]
    water_image = images[1][:, :, :3]
    cloud_image = images[2][:, :, :3]
    chl_image = images[3][:, :, :3]

    image_shape = true_image.shape
    transformed_image = np.zeros(image_shape, dtype=true_image.dtype)

    water_pixels_mask = (
        (water_image[:, :, 2] > water_image[:, :, 0])
        & (water_image[:, :, 2] > water_image[:, :, 1])
        )


    cloud_pixels_mask = (
        (cloud_image[:, :, 0] > cloud_image[:, :, 1])  # El canal rojo es mayor que el verde
        & (cloud_image[:, :, 0] > cloud_image[:, :, 2])  # El canal rojo es mayor que el azul
        & (cloud_image[:, :, 0] > 190)  # El canal rojo es mayor que 190
    )

    transformed_image[water_pixels_mask] = [0, 0, 255]
    transformed_image[cloud_pixels_mask] = [255, 255, 255]

    white_pixels_mask = np.all(chl_image[:, :, :3] == [255, 255, 255], axis=-1)
    black_pixels_mask = np.all(chl_image[:, :, :3] == [0, 0, 0], axis=-1)

    non_black_or_white_mask = ~white_pixels_mask & ~black_pixels_mask

    chl_mask = water_pixels_mask & ~cloud_pixels_mask & non_black_or_white_mask
    transformed_image[chl_mask] = chl_image[chl_mask]
    return transformed_image

def calculate_mci(concentrations_areas):
    concentration_values = np.array([1, 2, 4, 8, 16])
    mci = np.dot(concentration_values, concentrations_areas)
    return mci

