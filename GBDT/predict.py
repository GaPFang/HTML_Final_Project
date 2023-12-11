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
argv1 = int(sys.argv[1])

y_svm, X_svm = svm_read_problem('train.txt')
X_svm = np.array(X_svm)
y_svm = np.array(y_svm)
X = []
y = []
for i in range(len(X_svm)):
    X.append([])
    for j in range(len(X_svm[i])):
        X[i].append(X_svm[i][j + 1])
    y.append(y_svm[i])

n_estimators = 1
learning_rate = 0.1
max_depth = 10

# Create a GradientBoostingRegressor

params = {
    'n_estimators': n_estimators,  # Number of boosting stages to be run
    'learning_rate': learning_rate,  # Step size shrinkage used to prevent overfitting
    'max_depth': max_depth,  # Maximum depth of the individual trees
    'random_state': (argv1 + 1) * (argv1 + 2),  # Seed for reproducibility
}

gb_regressor = GradientBoostingRegressor(**params)
gb_regressor.fit(X, y)
dirName = "models_" + str(n_estimators) + "_" + str(learning_rate) + "_" + str(max_depth)

#################################################################################################

date = ["20231021", "20231022", "20231023", "20231024", "20231211", "20231212", "20231213", "20231214", "20231215", "20231216", "20231217"] 
models = os.listdir(dirName)
time_slice = 20

stationList = []
f = open("./html.2023.final.data/sno_test_set.txt", "r")
for line in f:
    stationList.append(line[:-1])
f.close()

actStationList = []
totList = []
f = open("./stationList.txt", "r")
for line in f:
    actStationList.append(line[:9])
    totList.append(int(line[10:-1]))
f.close()

pos = []
f = open("./html.2023.final.data/demographic.json", "r")
data = json.load(f)
for station in stationList:
    pos.append([data[station]['lat'], data[station]['lng']])
f.close()

data = []
tot_count = 0
for i in range(len(stationList)):
    if stationList[i] not in actStationList:
        for d in date:
            for j in range(72):
                data.append([d + "_" + stationList[i] + "_" + str(minToTime(j * 20)), 0])
        continue
    x_all = []
    for d in date:
        week = calWeek(d)
        for k in range(72):
            x = []
            x.append(week)
            x.append(totList[tot_count])
            min = k * 20
            for j in range(time_slice):
                x.append(pow(1 - abs(min - 1440 / time_slice * j) / 1440, 3))
            x.append(pos[i][0])
            x.append(pos[i][1])
            x_all.append(x)
        y_pred = gb_regressor.predict(x_all)
    for d in date:
        for j in range(72):
            index = date.index(d) * 72 + j
            sbi = float(y_pred[index])
            if sbi < 0:
                sbi = 0
            elif sbi > totList[tot_count]:
                sbi = totList[tot_count]
            data.append([d + "_" + stationList[i] + "_" + str(minToTime(j * 20)), float(sbi)])
    tot_count += 1
if "predicts" not in os.listdir("./"):
    os.mkdir("./predicts")
outfile = "./predicts/GBDT_" + str(n_estimators) + "_" + str(learning_rate) + "_" + str(max_depth) + "_" + str(argv1) + ".csv"
with open(outfile, 'w', newline='') as submissions:
    writer = csv.writer(submissions)
    writer.writerow(['id', 'sbi'])
    writer.writerows(data)

if dirName not in os.listdir("./"):
    os.mkdir(dirName)
model_name = dirName + "/GBDT_" + str(argv1) + ".model"
joblib.dump(gb_regressor, model_name)