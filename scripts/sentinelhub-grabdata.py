import os
import numpy as np
import matplotlib.pyplot as plt
from sentinelhub import (
    SHConfig,
    CRS,
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


config = SHConfig()

if not config.sh_client_id or not config.sh_client_secret:
    print("Warning! To use Process API, please provide the credentials (OAuth client ID and client secret).")
    
sukhna_coords_wgs84 = (76.8, 30.73, 76.84, 30.75)
resolution = 5
sukhna_bbox = BBox(bbox=sukhna_coords_wgs84, crs=CRS.WGS84)
sukhna_size = bbox_to_dimensions(sukhna_bbox, resolution=resolution)

print(f"Image shape at {resolution} m resolution: {sukhna_size} pixels")

# Read the JavaScript file
current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
evalscripts = os.path.join(parent_directory, 'evalscripts')
dat_folder = os.path.join(parent_directory, 'data')

evalscript_water_path = os.path.join(evalscripts, 'true-color.js')
with open(evalscript_water_path, 'r') as file:
    evalscript_water = file.read()

# Define time intervals for each month in a year
time_intervals = [
    ("2020-01-01", "2020-01-31"),
    ("2020-02-01", "2020-02-29"),
    ("2020-03-01", "2020-03-31"),
    ("2020-04-01", "2020-04-30"),
    ("2020-05-01", "2020-05-31"),
    ("2020-06-01", "2020-06-30"),
    ("2020-07-01", "2020-07-31"),
    ("2020-08-01", "2020-08-31"),
    ("2020-09-01", "2020-09-30"),
    ("2020-10-01", "2020-10-31"),
    ("2020-11-01", "2020-11-30"),
    ("2020-12-01", "2020-12-31"),
]

# Loop through each time interval and request images
for time_interval in time_intervals:
    request_true_color = SentinelHubRequest(
        evalscript=evalscript_water,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.PNG)],
        bbox=sukhna_bbox,
        size=sukhna_size,
        config=config,
    )

    true_color_imgs = request_true_color.get_data()
    image = true_color_imgs[0]
    start_date = time_interval[0]
    image_path = os.path.join(dat_folder, f'sukhna_lake_true-color_{start_date}.npy')
    np.save(image_path, image)
    
    print (f"Image saved to {image_path}")