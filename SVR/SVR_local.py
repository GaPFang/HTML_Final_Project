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

C = 0.1
gamma = 100
epsilon =  0.1

f = open("/dev/null", "w")
os.dup2(f.fileno(), sys.stdout.fileno())
f.close()

stationList = []
f = open("./stationList.txt", "r")
for line in f:
    stationList.append(line[:9])
f.close()

x_all = []
y_all = []
for i in range(len(stationList)):
    x_all.append([])
    y_all.append([])
    path = "./train/train_" + stationList[i] + ".txt"
    y_all[i], x_all[i] = svm_read_problem(path)

if os.path.exists("models") == False:
    os.mkdir("models")

for l in range(len(stationList)):
    N = len(x_all[l])
    param = svm_parameter('-s 3 -t 2 -c ' + str(C) + ' -g ' + str(gamma) + ' -p ' + str(epsilon) + ' -q')
    prob = svm_problem(y_all[l], x_all[l])
    m = svm_train(prob, param)
    svm_save_model("models/" + "model" + stationList[l] + ".model", m)