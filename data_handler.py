import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

'''
weather_df: 20-minute weather data using linear method (source: weather/visualcrossing).
station_20_dfs: A diction whose keys are stations. value: df that is 20-min average. 
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basically handle some data.')
    parser.add_argument('--weather_file', default='weather/visualcrossing/merged_20231001_20231128_20231211.csv', help='A weather file from visualcrossing')
    args = parser.parse_args()

    weather_df = pd.read_csv(args.weather_file)
    weather_df['datetime'] = pd.to_datetime(weather_df['datetime'])
    weather_df.drop(['stations', 'icon', 'name', 'conditions'], axis=1, inplace=True)
    weather_df['preciptype'].replace({'rain': 1, 'N/A': 0}, inplace=True)
    helper = pd.DataFrame({'datetime': pd.date_range(start=weather_df['datetime'].min(), end=weather_df['datetime'].max(),freq='20T')})
    weather_df = pd.merge(weather_df, helper, on='datetime', how='outer').sort_values('datetime')
    weather_df.set_index('datetime', inplace=True)
    weather_df = weather_df.interpolate(method='linear')
    # print(weather_df)

    ntu_stations = []
    with open('html.2023.final.data/sno_test_set.txt', 'r') as file:
        for line in file:
            # Remove any whitespace characters (like newline) and append to the list
            ntu_stations.append(line.strip())

    # Load station bike data
    stations_data = {station: {} for station in ntu_stations}

    data_directory = 'html.2023.final.data/release/'
    date_directory = [d for d in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, d))]
    date_directory.sort()

    ii = 3
    for date in tqdm(date_directory):
        if ii == 0:
            break
        ii -= 1
        dir_path = os.path.join(data_directory, date)
        if not os.path.isdir(dir_path):
            continue
        for station in tqdm(ntu_stations, leave=False):
            station_file = station+'.json'
            json_data = json.load(open(os.path.join(dir_path, station_file), 'r'))
            for time_slot, slot_data in json_data.items():
                combined_timestamp = pd.to_datetime(f"{date} {time_slot}", format="%Y%m%d %H:%M")
                if combined_timestamp not in stations_data[station]:
                    stations_data[station][combined_timestamp] = {}
                if len(slot_data) == 0:
                    stations_data[station][combined_timestamp]['tot'] = np.nan
                    stations_data[station][combined_timestamp]['sbi'] = np.nan
                    stations_data[station][combined_timestamp]['bemp'] = np.nan
                    continue
                stations_data[station][combined_timestamp]['tot'] = slot_data['tot']
                stations_data[station][combined_timestamp]['sbi'] = slot_data['sbi']
                stations_data[station][combined_timestamp]['bemp'] = slot_data['bemp']

    # Create DataFrame from the dictionary
    station_dfs = {station: pd.DataFrame.from_dict(stations_data[station], orient='index') for station in ntu_stations}
    station_20_dfs = {}
    for station in tqdm(ntu_stations):
        station_20_dfs[station] = station_dfs[station].rolling('20T', center=True).mean().resample('20T').first()
        station_20_dfs[station]['hour'] = station_20_dfs[station].index.hour
        station_20_dfs[station]['minute'] = station_20_dfs[station].index.minute
        station_20_dfs[station]['day_of_week'] = station_20_dfs[station].index.dayofweek
    print(station_20_dfs['500101001'])