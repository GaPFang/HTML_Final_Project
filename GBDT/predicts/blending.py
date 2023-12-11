from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from libsvm.svmutil import *
import sys
import time
import csv
import json
import os
import joblib

def minToTime(min):
    return str(int(min / 60)).zfill(2) + ":" + str(min % 60).zfill(2)

def calWeek(date):
    month = int(date[4:6])
    day = int(date[6:8])
    if month == 11:
        day += 31
    elif month == 12:
        day += 61
    return (day + 6) % 7

csvList = os.listdir('.')

data = np.zeros((88705, 2)).tolist()
data[0] = ['id', 'sbi']

# average csv files
for csvFile in csvList:
    if csvFile[-4:] != '.csv':
        continue
    with open(csvFile, newline='') as csvfile:
        rows = csv.reader(csvfile)
        row = next(rows)
        i = 1
        for row in rows:
            data[i][0] = row[0]
            data[i][1] += float(row[1])
            i += 1

for i in range(1, 88705):
    data[i][1] /= 7

# write average csv file
with open('blending.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)