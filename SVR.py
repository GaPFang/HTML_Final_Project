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

C = [0.1, 1, 10, 100, 1000]
gamma = [0.01, 0.1, 1, 10, 100]
epsilon =  [0.01, 0.1, 1, 10, 100]

f = open("/dev/null", "w")
os.dup2(f.fileno(), sys.stdout.fileno())
f.close()

def calError(y_hat, y, x):
    err = 0
    for i in range(len(y)):
        err += 3 * abs((y_hat[i] - y[i]) / x[i][3]) * (abs(y_hat[i] / x[i][3] - 1 / 3) + abs(y_hat[i] / x[i][3] - 2 / 3))
    return err / len(y)

fold = 10
repeat = 10

stationList = []
f = open("./stationList.txt", "r")
for line in f:
    stationList.append(line[:-1])
f.close()

x_all = []
y_all = []
for i in range(len(stationList)):
    x_all.append([])
    y_all.append([])
    path = "./train/train_" + stationList[i] + ".txt"
    y_all[i], x_all[i] = svm_read_problem(path)

# i = int(input("index: ")) - 1

if os.path.exists("models") == False:
    os.mkdir("models")
# proFd = open("progress.txt", "w")
# resultFd = open("result" + str(i + 1) + ".txt", "w")
# resultFd = open("result.txt", "w")
# read_pipe = np.zeros((len(C), len(gamma), len(epsilon))).tolist()
count = 0
index = int(sys.argv[1]) - 1
for l in range(math.ceil(index * len(stationList) / 5), math.ceil((index + 1) * len(stationList) / 5)):
    pid = os.fork()
    N = len(x_all[l])
    count += 1
    if pid == 0:
        result = {}
        E_CV = np.zeros((len(C), len(gamma), len(epsilon))).tolist()
        for i in range(len(C)):
            for j in range(len(gamma)):
                for k in range(len(epsilon)):
                    param = svm_parameter('-s 3 -t 2 -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k]) + ' -q')
                    for r in range(repeat):
                        np.random.seed(np.random.seed(r))
                        mess = np.random.permutation(N).tolist()
                        folds = []
                        for v in range(fold):
                            folds.append(mess[int(v * N / fold):int((v + 1) * N / fold)])
                        for v in range(fold):
                            x_train = []
                            y_train = []
                            x_val = []
                            y_val = []
                            for a in range(fold):
                                if a == v:
                                    for b in range(int(N/fold)):
                                        x_val.append(x_all[l][folds[a][b]])
                                        y_val.append(y_all[l][folds[a][b]])
                                else:
                                    for b in range(int(N/fold)):
                                        x_train.append(x_all[l][folds[a][b]])
                                        y_train.append(y_all[l][folds[a][b]])
                            prob = svm_problem(y_train, x_train)
                            m = svm_train(prob, param)
                            p_label, p_acc, p_val = svm_predict(y_val, x_val, m)        
                            E_CV[i][j][k] += calError(p_label, y_val, x_val)
                    E_CV[i][j][k] /= (fold * repeat)
                    result.update({E_CV[i][j][k]: [C[i], gamma[j], epsilon[k]]})
        result = OrderedDict(sorted(result.items()))
        max_key_length = max(len(str(key)) for key in result)
        f = open("models/model_500101001.txt", "w")
        for key in result:
            f.write(str(result[key][0]) + " " + str(result[key][1]) + " " + str(result[key][2]) + " E_CV: " + str(r).ljust(max_key_length) + "\n")
        f.close()
        os._exit(0)
    if count >= 30:
        os.wait()
        count -= 1
        