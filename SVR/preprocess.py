import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import json

time_slice = 20

def calWeek(date):
    month = int(date[4:6])
    day = int(date[6:8])
    if month == 11:
        day += 31
    elif month == 12:
        day += 61
    return (day + 6) % 7

stationList = []
# f = open("./html.2023.final.data/demographic.json", "r")
# data = json.load(f)
# for station in data:
#     stationList.append(station)
# f.close()
f = open("./html.2023.final.data/sno_test_set.txt", "r")
for line in f:
    stationList.append(line[:-1])
f.close()

train_x = []
train_y = []
dateList = os.listdir("./html.2023.final.data/release/")
for i in range(len(stationList)):
    train_x.append([])
    train_y.append([])
    for date in dateList:
        week = calWeek(date)
        path = "./html.2023.final.data/release/" + date + "/" + stationList[i] + ".json"
        if os.path.exists(path) == False:
            continue
        f = open(path, "r")
        data = json.load(f)
        f.close()
        flag = False
        for time in data:
            if int(time[3:]) % 10 == 0:
                flag = False
            if flag:
                continue
            if data[time] != {}:
                flag = True
                if int(data[time]['act']) == 0:
                    continue
                x = {}
                x.update({1: week})
                x.update({2: int(data[time]['tot'])})
                min = int(time[:2]) * 60 + int(time[3:])
                for j in range(time_slice):
                    x.update({3 + j: pow(1 - abs(min - 1440 / time_slice * j) / 1440, 3)})
                train_x[i].append(x)
                train_y[i].append(int(data[time]['sbi']))

if os.path.exists("train") == False:
    os.mkdir("train")
f = []
for i in range(len(stationList)):
    f.append(open("train/train_" + stationList[i] + ".txt", "w"))
for i in range(len(train_x)):
    for j in range(len(train_x[i])):
        f[i].write(str(train_y[i][j]) + " ")
        for k in train_x[i][j]:
            f[i].write(str(k) + ":" + str(train_x[i][j][k]) + " ")
        f[i].write("\n")

stationFd = open("./stationList.txt", "w")
for i in range(len(stationList)):
    f[i].close()
    if os.path.getsize("train/train_" + stationList[i] + ".txt") == 0:
        os.remove("train/train_" + stationList[i] + ".txt")
    else :
        stationFd.write(stationList[i] + " " + str(train_x[i][0][2]) + "\n")
stationFd.close()