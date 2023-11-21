import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import json

def calWeek(date):
    month = int(date[4:6])
    day = int(date[6:8])
    if month == 11:
        day += 31
    elif month == 12:
        day += 61
    return (day + 6) % 7

stationList = []
f = open("./html.2023.final.data/sno_test_set.txt", "r")
for line in f:
    stationList.append(line.strip())
f.close()

i = 0
train_x = []
train_y = []
for i in range(len(stationList)):
    train_x.append([])
    train_y.append([])
dateList = os.listdir("./html.2023.final.data/release/")
for date in dateList:
    week = calWeek(date)
    for i in range(len(stationList)):
        f = open("./html.2023.final.data/release/" + date + "/" + stationList[i] + ".json", "r")
        data = json.load(f)
        f.close()
        flag = False
        for time in data:
            if int(time[3:]) % 20 == 0:
                flag = False
            if flag:
                continue
            if data[time] != {}:
                x = {}
                x.update({1: int(date)})
                x.update({2: int(time[:2]) * 60 + int(time[3:])})
                x.update({3: int(data[time]['tot'])})
                x.update({4: int(data[time]['act'])})
                x.update({5: week})
                train_x[i].append(x)
                train_y[i].append(int(data[time]['sbi']))
                flag = True

f = []
for i in range(len(stationList)):
    f.append(open("train/train_" + stationList[i] + ".txt", "w"))
for i in range(len(train_x)):
    for j in range(len(train_x[i])):
        f[i].write(str(train_y[i][j]) + " ")
        for k in train_x[i][j]:
            f[i].write(str(k) + ":" + str(train_x[i][j][k]) + " ")
        f[i].write("\n")
for i in range(len(stationList)):
    f[i].close()
        


# sbi_20min = []
# act_20min = []
# tot_20min = []
# for i in range(len(sbi_all)):
#     sbi_20min.append([])
#     act_20min.append([])
#     tot_20min.append([])
#     for j in range(len(sbi_all[i])):
#         sbi_20min[i].append([])
#         act_20min[i].append([])
#         tot_20min[i].append([])
#         for k in range(72):
#             sbi_20min[i][j].append(sum(sbi_all[i][j][k:k + 20]) / 20)
#             act_20min[i][j].append(sum(act_all[i][j][k:k + 20]) / 20)
#             tot_20min[i][j].append(sum(tot_all[i][j][k:k + 20]) / 20)
# print(len(sbi_20min))
# print(len(sbi_20min[0]))
# print(sbi_20min)


        

                





