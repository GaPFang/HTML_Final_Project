import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
from tqdm import tqdm
import json
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
# from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

def day_pass(date: str, stdandard: str = '20231001'):
    return (datetime.strptime(date, '%Y%m%d')-datetime.strptime(stdandard, '%Y%m%d')).days

def normalize(station_data: dict):
    # normalize the data
    norm_data = {key: np.array(values) for key, values in station_data.items()}
    mean = {key: np.mean(values) for key, values in norm_data.items()}
    min = {key: np.max(values) for key, values in norm_data.items()}
    max = {key: np.min(values) for key, values in norm_data.items()}
    norm_data = {key: [(x-mean[key])/(max[key]-min[key]) for x in values] for key, values in norm_data.items()}
    return norm_data

def shuffle(data, length, seed=1509):
    np.random.seed(seed)
    shuffled_indices = np.random.permutation(len(data['lat']))
    shuffled_data = {key: [values[i] for i in shuffled_indices] for key, values in data.items()}
    return shuffled_data


if __name__ == "__main__":
    # ref: https://daniel820710.medium.com/%E5%88%A9%E7%94%A8keras%E5%BB%BA%E6%A7%8Blstm%E6%A8%A1%E5%9E%8B-%E4%BB%A5stock-prediction-%E7%82%BA%E4%BE%8B-1-67456e0a0b
    
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

    station_data = {'lat': [], 'lng': [], 'date':[], 'day': [], 'time': [], 'sbi': [], 'tot': []}

    ii = 3
    for date in tqdm(date_directory):
        if ii == 0:
            break
        ii -= 1
        dir_path = os.path.join(data_directory, date)
        if not os.path.isdir(dir_path):
            continue
        station_files = sorted(os.listdir(dir_path))
        jj = 5
        for station_file in tqdm(station_files, leave=False):
            if jj == 0:
                break
            jj -= 1
            station = os.path.splitext(station_file)[0]
            json_data = json.load(open(os.path.join(dir_path, station_file), 'r'))
            day = (datetime.strptime(date, '%Y%m%d')).weekday()
            for time_slot, slot_data in json_data.items():
                if(len(slot_data) == 0):
                    continue
                time = int(time_slot.split(':')[0]) * 60 + int(time_slot.split(':')[1])
                station_data['lat'].append(position[station][0])
                station_data['lng'].append(position[station][1])
                station_data['day'].append(day)
                station_data['date'].append(day_pass(date))
                station_data['time'].append(time)
                station_data['tot'].append(slot_data['tot'])
                station_data['sbi'].append(slot_data['sbi'])

    # norm_data = normalize(station_data)

    # shuffled_data = shuffle(station_data, len(station_data['lat']))

    X = [value for key, value in list(station_data.items())[:-1]]
    X = [list(t) for t in zip(*X)]
    Y = station_data['sbi']
    X_train = X[:int(0.8*len(X))]
    X_test = X[int(0.8*len(X)):]
    Y_train = Y[:int(0.8*len(X))]
    Y_test = Y[int(0.8*len(X)):]

    model = Sequential()
    # Input layer with 6 neurons (corresponding to 6D input)
    model.add(Dense(12, input_dim=6, activation='relu'))
    # model.add(Dense(8, activation='relu'))  # Hidden layer with 8 neurons
    # # Output layer with 1 neuron for regression (linear activation)
    # model.add(Dense(1, activation='linear'))

    # model.compile(loss='mse', optimizer='adam', metrics=['mae'])
    # # model.compile(optimizer='rmsprop', loss='mse')
    # model.fit(X_train, Y_train, epochs=50, batch_size=64, validation_data=(X_test, Y_test), verbose=1)
    # model.save('LSTM.h5')