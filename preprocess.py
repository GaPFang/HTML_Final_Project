# similar to Ming but a little different
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

sbi_all = {key: [[[] for _ in range(24*60)] for _ in range(7)] for key in list(position)}
tot_all = {key: [[[] for _ in range(24*60)] for _ in range(7)] for key in list(position)}

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
            sbi_all[station][day][time].append(slot_data['sbi'])
            tot_all[station][day][time].append(slot_data['tot'])
print("calculating means")

sbi_mean = {key: [[0 for j in range(24*60)] for i in range(7)] for key, value in sbi_all.items()}
for station in list(sbi_all):
    for i in range(7):
        for j in range(24*60):
            if len(sbi_all[station][i][j]) != 0:
                sbi_mean[station][i][j] = np.mean(sbi_all[station][i][j])

print("means have been calculated")
