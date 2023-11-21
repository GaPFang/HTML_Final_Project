import json
import numpy as np
import math
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# demographic

# Read the JSON file
file_path = 'html.2023.final.data/demographic.json'
with open(file_path, 'r', encoding='utf-8') as file:
    json_data = json.load(file)

# Create a dictionary with IDs as keys and lat/lng as values
position = {key: [value["lat"], value["lng"]] for key, value in json_data.items()}

# Directory where your data is stored
data_directory = 'html.2023.final.data/release/'

# Get a list of directories in the specified path
date_directory = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
date_directory.sort()

station_data = {key: [[[] for _ in range(24*60)] for _ in range(7)] for key in list(position)}

for date in tqdm(date_directory):
    # for date in date_directory:
    dir_path = os.path.join(data_directory, date)
    if not os.path.isdir(dir_path):
        continue
    station_files = sorted(os.listdir(dir_path))
    for station_file in tqdm(station_files, leave=False):
    # for station_file in station_files:
        station = os.path.splitext(station_file)[0]
        json_data = json.load(open(os.path.join(dir_path, station_file), 'r'))
        day = (datetime.strptime(date, '%Y%m%d')).weekday()
        for time_slot, slot_data in json_data.items():
            if len(slot_data) == 0:
                continue
            time = int((datetime.strptime(time_slot, '%H:%M') - datetime.strptime('00:00', '%H:%M')).total_seconds() / 60)
            station_data[station][day][time].append(slot_data['sbi'])

station_mean = {key: [[0 for j in range(24*60)] for i in range(7)] for key, value in station_data.items()}
for station in list(station_data):
    for i in range(7):
        for j in range(24*60):
            if len(station_data[station][i][j]) != 0:
                station_mean[station][i][j] = np.mean(station_data[station][i][j])

# station_mean = {key: [[np.mean(value[i][j]) for j in range(24*60)] for i in range(7)] for key, value in station_data.items()}

print(station_mean['500101001'][0])

for station, data in tqdm(station_mean.items()):
    with open('baseline_'+station, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(station_mean[station])
