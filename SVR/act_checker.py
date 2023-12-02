import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import json

dateList = os.listdir("./html.2023.final.data/release/")
for date in dateList:
    stationList = os.listdir("./html.2023.final.data/release/" + date)
    for station in stationList:
        f = open("./html.2023.final.data/release/" + date + "/" + station)
        data = json.load(f)
        f.close()
        for time in data:
            if data[time] != {} and data[time]['act'] != '1':
                print(date, station, time)



