import json
import numpy as np
import math
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

predict_days = [(datetime(2023, 10, 21) + timedelta(days=i)).strftime('%Y%m%d') for i in range(4)] + [(datetime(2023, 12, 4) + timedelta(days=i)).strftime('%Y%m%d') for i in range(7)]

with open('baseline_exported.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'sbi'])
    for date in tqdm(predict_days):
        day = (datetime.strptime(date, '%Y%m%d')).weekday()
        with open('sno_test_set.txt', 'r') as file:
            for station in tqdm(file, leave=False):
                station = station.replace('\n', '')
                with open('baseline_data/baseline_'+station+'.csv', newline='') as pred:
                    csvreader = csv.reader(pred)
                    rows = list(csvreader)
                    for t in range(24*3):
                        time = (datetime.strptime(date, "%Y%m%d")+timedelta(minutes=t*20)).strftime('%H:%M')
                        if t == 0:
                            predict = np.mean(list(map(float, rows[day][0:11])))
                        else:
                            predict = np.mean(list(map(float, rows[day][t*20-10:t*20+11])))
                        id = date+'_'+station+'_'+time
                        writer.writerow([id, predict])
