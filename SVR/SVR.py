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

C = [0.01, 0.1, 1, 5, 10]
gamma = [0.0005, 0.005, 0.05, 0.5, 5]
epsilon =  [0.01, 0.1, 1, 10, 100]

f = open("/dev/null", "w")
os.dup2(f.fileno(), sys.stdout.fileno())
f.close()

def calError(y_hat, y, x):
    err = 0
    for i in range(len(y)):
        # if x[i][3] == 0:
        #     return -1
        err += 3 * abs((y_hat[i] - y[i]) / x[i][2]) * (abs(y_hat[i] / x[i][2] - 1 / 3) + abs(y_hat[i] / x[i][2] - 2 / 3))
    return err / len(y)

fold = 10
repeat = [3, 10]

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
# for l in range(len(stationList)):
    pid = os.fork()
    N = len(x_all[l])
    count += 1
    if pid == 0:
        result = {}
        E_CV = np.zeros((len(C), len(gamma), len(epsilon))).tolist()
############################___________1___________############################
        for i in range(len(C)):
            for j in range(len(gamma)):
                for k in range(len(epsilon)):
                    param = svm_parameter('-s 3 -t 2 -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k]) + ' -q')
                    np.random.seed(int(time.time()))
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
                        # if calError(p_label, y_val, x_val) == -1:
                        #     print("station: " + stationList[i], file=sys.stderr)
                        #     print(str(x_val), file=sys.stderr)
                    E_CV[i][j][k] /= fold
                    result.update({E_CV[i][j][k]: [i, j, k]})
        result = OrderedDict(sorted(result.items()))
############################___________2___________############################
        count2 = 0
        for key in result:
            if count2 >= 8:
                break
            count2 += 1
            result2 = {}
            i, j, k = result[key]
            param = svm_parameter('-s 3 -t 2 -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k]) + ' -q')
            for re in range(repeat[0]):
                E = 0
                np.random.seed(int(time.time()))
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
                    E += calError(p_label, y_val, x_val)
                E /= fold
                E_CV[i][j][k] *= E
            result2.update({E_CV[i][j][k]: [i, j, k]})
            result2 = OrderedDict(sorted(result2.items()))
############################___________3___________############################
        count3 = 0
        for key2 in result2:
            if count3 >= 3:
                break
            count3 += 1
            result3 = {}
            i, j, k = result2[key2]
            param = svm_parameter('-s 3 -t 2 -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k]) + ' -q')
            for re in range(repeat[1]):
                E = 0
                np.random.seed(int(time.time()))
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
                    E += calError(p_label, y_val, x_val)
                E /= fold
                E_CV[i][j][k] *= E
############################___________END___________############################
            E_CV[i][j][k] = pow(E_CV[i][j][k], 1 / (1 + repeat[0] + repeat[1]))
            result3.update({E_CV[i][j][k]: [i, j, k]})
            result3 = OrderedDict(sorted(result3.items()))
        max_key_length = max(len(str(key)) for key in result3)
        f = open("models/model" + str(stationList[l]) + ".txt", "w")
        for key in result3:
            f.write(str(C[result3[key][0]]) + " " + str(gamma[result3[key][1]]) + " " + str(epsilon[result3[key][2]]) + " E_CV: " + str(key).ljust(max_key_length) + "\n")
        f.close()
        prob = svm_problem(y_all[l], x_all[l])
        param = svm_parameter('-s 3 -t 2 -c ' + str(C[result3[key][0]]) + ' -g ' + str(gamma[result3[key][1]]) + ' -p ' + str(epsilon[result3[key][2]]) + ' -q')
        m = svm_train(prob, param)
        svm_save_model("models/model" + str(stationList[l]) + ".model", m)
        os._exit(0)
    if count >= 30:
        os.wait()
        count -= 1
while count > 0:
    os.wait()
    count -= 1
        