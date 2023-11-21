import json
import numpy as np
import math
from datetime import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations_with_replacement
import pandas as pd
import plotly.express as px
import pmdarima as pm

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

# station_data = {key: {'time': [], 'sbi': [], 'tot': [], 'ratio': []} for key in list(position)}
hour_data = {key: {'time': [], 'ratio': []} for key in list(position)}

for date in tqdm(date_directory):
    dir_path = os.path.join(data_directory, date)
    if not os.path.isdir(dir_path):
        continue
    station_files = sorted(os.listdir(dir_path))
    for station_file in tqdm(station_files, leave=False):
        station = os.path.splitext(station_file)[0]
        json_data = json.load(open(os.path.join(dir_path, station_file), 'r'))
        time_now = pd.to_datetime(date+' 00:00:00')
        time_interval = pd.Timedelta(minutes=15)
        tmp_list = []
        for time_slot, slot_data in json_data.items():
            time = pd.to_datetime(date+' '+time_slot)
            if(len(slot_data) != 0):
                # station_data[station]['time'].append(time)
                # station_data[station]['sbi'].append(slot_data['sbi'])
                # station_data[station]['tot'].append(slot_data['tot'])
                # station_data[station]['ratio'].append(slot_data['sbi']/slot_data['tot'])
                tmp_list.append(slot_data['sbi']/slot_data['tot'])
            if time - time_now == time_interval:
                hour_data[station]['time'].append(time_now)
                if len(tmp_list) == 0:
                    hour_data[station]['ratio'].append(0)
                else:
                    hour_data[station]['ratio'].append(np.mean(tmp_list))
                tmp_list = []
                time_now += time_interval
        break

# df = pd.DataFrame(hour_data['500101001'])
# fig = px.scatter(df, x='time', y='ratio')
# fig.write_image("scatter_plot.png")

# for station in list(position):
for station in ['500101001']:
    data = pd.DataFrame({'Timestamp': pd.to_datetime(hour_data[station]['time']), 'ratio': hour_data[station]['ratio']})
    data.set_index('Timestamp', inplace=True)
    train=data[(data.index.get_level_values(0) <= '2023-11-10')]
    valid=data[(data.index.get_level_values(0) > '2023-11-10')]
    # model = pm.auto_arima(data, m=96, seasonal=True, start_p=0, start_q=0, max_order=4, test='adf',error_action='ignore', suppress_warnings=True, stepwise=True, trace=True)
    model = pm.auto_arima(data, m=96, seasonal=True,error_action='ignore', trace=True)
    print("model parameter decided.")
    model_fit = model.fit(train)
    print("model fitted.")
    model_fit.save('arima_model_'+station+'.pkl')
    print("model saved.")
    forecast=model.predict(n_periods=9, return_conf_int=True)
    print("prediction completed.")
    forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])
    plt.plot(train, label='Train')
    plt.plot(valid, label='Valid')
    plt.plot(forecast, label='Prediction')
    plt.savefig('result.png')
# X = {key: station_data['time'] for key in station.keys()}
# Y = {key: station_data['ratio'] for key in station.keys()}

