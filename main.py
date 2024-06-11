from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import pyproj
import numpy as np
import dask
import dask.multiprocessing
from dask import delayed, compute
import time
import csv
import os


# Function to read TLE data from file
def read_tle_file(file_path):
    tle_data = []
    try:
        with open(file_path, 'r') as file:
            lines = iter(file)
            for satellite_name in lines:
                line1 = next(lines).strip()
                line2 = next(lines).strip()
                tle_data.append((satellite_name.strip(), line1, line2))
        print(f"Read {len(tle_data)} satellites from TLE file.")
    except Exception as e:
        print(f"Error reading TLE file: {e}")
    return tle_data

# Initialize the Transformer object outside the function to avoid re-initializing it multiple times.
ecef = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
lla = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
transformer = pyproj.Transformer.from_proj(ecef, lla)

# Function to convert ECEF coordinates to latitude, longitude, and altitude
def ecef2lla(pos_x, pos_y, pos_z):
    lon, lat, alt = transformer.transform(pos_x, pos_y, pos_z, radians=False)
    return lon, lat, alt


# Function to calculate satellite positions for a chunk of data
@delayed
def calculate_satellite_positions_chunk(chunk, start_time, end_time, interval_minutes):
    satellite_positions = []
    current_time = start_time
    while current_time <= end_time:
        year, month, day, hour, minute, second = (
            current_time.year, current_time.month, current_time.day,
            current_time.hour, current_time.minute, current_time.second
        )
        jd, fr = jday(year, month, day, hour, minute, second)

        for satellite_data in chunk:
            satellite_name, line1, line2 = satellite_data
            sat = Satrec.twoline2rv(line1, line2)
            e, r, vel = sat.sgp4(jd, fr)
            lon, lat, alt = ecef2lla(r[0], r[1], r[2])
            satellite_positions.append((satellite_name, current_time, lon, lat, alt, vel))
        current_time += timedelta(minutes=interval_minutes)
    return satellite_positions

# Function to divide the list of satellites into chunks
def chunkify(lst, num_chunks):
    chunk_size = max(1, len(lst) // num_chunks)
    chunks = [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]
    return chunks

# Function to check if a point is within the defined rectangular region
def is_within_region(lon, lat, region_coords):
    lon_min = min(coord[1] for coord in region_coords)
    lon_max = max(coord[1] for coord in region_coords)
    lat_min = min(coord[0] for coord in region_coords)
    lat_max = max(coord[0] for coord in region_coords)
    return lon_min <= lon <= lon_max and lat_min <= lat <= lat_max

# Main function to perform parallel processing with Dask
def main():
    file_path = '30sats.txt'
    tle_data = read_tle_file(file_path)
    start_time = datetime(2023, 1, 1)
    end_time = datetime(2023, 1, 2)
    num_chunks = os.cpu_count() # Number of CPU cores available
    satellite_chunks = chunkify(tle_data, num_chunks)

    # Created tasks for all chunks. interval_minutes = 1. 
    tasks = [calculate_satellite_positions_chunk(chunk, start_time, end_time, 1) for chunk in satellite_chunks]
    results = compute(*tasks, scheduler='processes')

    satellite_positions = [position for result in results for position in result]

    # Print all results on console. 
    for satellite_position in satellite_positions:
        satellite_name, current_time, lon, lat, alt, vel = satellite_position
        print("Satellite:", satellite_name)
        print("Time:", current_time)
        print("Longitude:", lon)
        print("Latitude:", lat)
        print("Altitude:", alt)
        print("Velocity:", vel)
        print()

    ## Below code can be used to write all results on the file "satellite_positions.csv".
    # with open('satellite_positions.csv', mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['Satellite', 'Time', 'Longitude', 'Latitude', 'Altitude', 'Velocity'])
    #     for satellite_position in satellite_positions:
    #         satellite_name, current_time, lon, lat, alt, vel = satellite_position
    #         writer.writerow([satellite_name, current_time, lon, lat, alt, vel])


    # Get user-defined rectangular region
    region_coords = []
    for i in range(4):
        lat = float(input(f"Enter latitude for corner {i + 1}: "))
        lon = float(input(f"Enter longitude for corner {i + 1}: "))
        region_coords.append((lat, lon))

    # Filter satellite positions within the user-defined region
    filtered_positions = [
        position for position in satellite_positions
        if is_within_region(position[2], position[3], region_coords)
    ]

    # Printing filtered results on the console. 
    for satellite_position in filtered_positions:
        satellite_name, current_time, lon, lat, alt, vel = satellite_position
        print("Satellite:", satellite_name)
        print("Time:", current_time)
        print("Longitude:", lon)
        print("Latitude:", lat)
        print("Altitude:", alt)
        print("Velocity:", vel)
        print()


if __name__ == '__main__':
    main()
