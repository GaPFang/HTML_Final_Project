import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import json

dict = {}

f = open("./html.2023.final.data/demographic.json", "r")
demographic = json.load(f)
f.close()

def calWeek(date):
    month = int(date[4:6])
    day = int(date[6:8])
    if month == 11:
        day += 31
    elif month == 12:
        day += 61
    return (day + 6) % 7

count = 0
for i in demographic:
    dict.update({count:i})
    count += 1

f = open("./html.2023.final.data/release/20231002/500101001.json", "r")
data = json.load(f)
f.close()

i = 0
sbi_all = []      # sbi      [date][station][time] = data
act_all = []      # action
tot_all = []      # total
dateList = os.listdir("./html.2023.final.data/release/")
for date in dateList:
    week = calWeek(date)
    stationList = os.listdir("./html.2023.final.data/release/" + date)
    j = 0
    sbi_all.append([])
    act_all.append([])
    tot_all.append([])
    for station in stationList:
        f = open("./html.2023.final.data/release/" + date + "/" + station)
        data = json.load(f)
        f.close()
        sbi_all[i].append([])
        act_all[i].append([])
        tot_all[i].append([])
        for time in data:
            if data[time] == {}:
                sbi_all[i][j].append(-1)
                act_all[i][j].append(-1)
                tot_all[i][j].append(-1)
            else:
                sbi_all[i][j].append(int(data[time]['sbi']))
                act_all[i][j].append(int(data[time]['act']))
                tot_all[i][j].append(int(data[time]['tot']))
        j += 1
    i += 1
last_sbi = 0
last_act = 0
last_tot = 0
for i in range(len(sbi_all) - 1, -1, -1):
    for j in range(len(sbi_all[i]) - 1, -1, -1):
        for k in range(len(sbi_all[i][j]) - 1, -1, -1):
            if sbi_all[i][j][k] == -1:
                sbi_all[i][j][k] = last_sbi
                act_all[i][j][k] = last_act
                tot_all[i][j][k] = last_tot
            else:
                last_sbi = sbi_all[i][j][k]
                last_act = act_all[i][j][k]
                last_tot = tot_all[i][j][k]

# 1:lat 2:lng 3:date 4:time 5:tot 6:act 7:week
train_x = []
train_y = []
for i in range(len(sbi_all)):
    for j in range(len(sbi_all[i])):
        for k in range(72):
            train_y.append(sbi_all[i][j][k * 20])
            x = {}
            x.update({1: float(demographic[dict[j]]['lat'])})
            x.update({2: float(demographic[dict[j]]['lng'])})
            x.update({3: int(dateList[i])})
            x.update({4: int(k * 20)})
            x.update({5: int(tot_all[i][j][k * 20])})
            x.update({6: int(act_all[i][j][k * 20])})
            x.update({7: calWeek(dateList[i])})
            train_x.append(x)

train = [train_y, train_x]
f = open("train.txt", "w")
f.write(str(train))
f.close()


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


        

                





