# SVR with two constraints: non-negative and smaller than tot

from libsvm.svmutil import *
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import json
import time
import math
import csv

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

date = ["20231021", "20231022", "20231023", "20231024", "20231204", "20231205", "20231206", "20231207", "20231208", "20231209", "20231210"]

f = open("/dev/null", "w")
os.dup2(f.fileno(), sys.stdout.fileno())
f.close()

def calError(y_hat, y, x):
    err = 0
    for i in range(len(y)):
        err += 3 * abs((y_hat[i] - y[i]) / x[i][3]) * (abs(y_hat[i] / x[i][3] - 1 / 3) + abs(y_hat[i] / x[i][3] - 2 / 3))
    return err / len(y)

stationList = []
f = open("./sno_test_set.txt", "r")
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

data = []
y_all = np.zeros(len(date) * 72).tolist()
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
        for j in range(72):
            x_all.append({1: week, 2: j * 20, 3: int(totList[tot_count])})
    p_label, p_acc, p_val = svm_predict(y_all, x_all, svm_load_model("models/model" + str(stationList[i]) + ".model"))
    for d in date:
        for j in range(72):
            index = date.index(d) * 72 + j
            sbi = float(p_label[index])
            if sbi < 0:
                sbi = 0
            elif sbi > totList[tot_count]:
                sbi = totList[tot_count]
            data.append([d + "_" + stationList[i] + "_" + str(minToTime(j * 20)), float(sbi)])
    tot_count += 1

with open('submission.csv', 'w', newline='') as submissions:
    writer = csv.writer(submissions)
    writer.writerow(['id', 'sbi'])
    writer.writerows(data)