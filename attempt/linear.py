import json
import numpy as np
import math
from datetime import datetime
import os
from liblinear.liblinearutil import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations_with_replacement

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

x = []
y = []

ii = 1
for date in tqdm(date_directory):
    if ii == 0:
       break
    ii -= 1
    dir_path = os.path.join(data_directory, date)
    if not os.path.isdir(dir_path):
        continue
    day = (datetime.strptime(date, '%Y%m%d')).weekday()
    day_list = [0]*7
    day_list[day] = 1
    for station_file in sorted(os.listdir(dir_path)):
        station = os.path.splitext(station_file)[0]
        json_data = json.load(open(os.path.join(dir_path, station_file), 'r'))
        for time_slot, slot_data in json_data.items():
            if(len(slot_data) == 0):
                continue
            bundle = position[station] + [slot_data["tot"]] + day_list
            x.append(bundle)
            y.append(slot_data["sbi"])

print("Data has loaded")
print("size of data: ", len(x))

Phi = x

print("Phi has calculated.")
print("size of Phi: ", len(Phi))

la = 0.1
seg_size = int(len(Phi)/5)
accum_acc = 0
ii = 1
for seg in tqdm(range(5)):
    if ii == 0:
       break
    ii -= 1
    prob = problem(y[:seg*seg_size], Phi[:seg*seg_size])
    C = 1/(2*la)
    param = parameter(f"-s 0 -c {C:.6f} -e 0.000001 -q")
    print("traning started")
    m = train(prob, param)
    print("traning finished")
    p_label, p_acc, p_val = predict(y[seg*seg_size: (seg+1)*seg_size], Phi[seg*seg_size: (seg+1)*seg_size], m, "-q")
    print(p_label)
