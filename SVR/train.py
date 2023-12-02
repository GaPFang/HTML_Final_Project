# SVR with two constraints: non-negative and smaller than tot

from libsvm.svmutil import *
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
import json

C = 1000
gamma = 0.01
epsilon =  0.01

def calError(y_hat, y, x):
    err = 0
    for i in range(len(y)):
        err += 3 * abs((y_hat[i] - y[i]) / x[i][2]) * (abs(y_hat[i] / x[i][2] - 1 / 3) + abs(y_hat[i] / x[i][2] - 2 / 3))
    return err / len(y)

fold = 10
repeat = 10

y_all, x_all = svm_read_problem("./train/train_500101001.txt")

proFd = open("progress.txt", "w")
resultFd = open("result.txt", "w")
param = svm_parameter('-s 3 -t 2 -c ' + str(C) + ' -g ' + str(gamma) + ' -p ' + str(epsilon) + ' -q')
E_CV = 0
N = len(x_all)
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
                x_val.append(x_all[folds[a][b]])
                y_val.append(y_all[folds[a][b]])
        else:
            for b in range(int(N/fold)):
                x_train.append(x_all[folds[a][b]])
                y_train.append(y_all[folds[a][b]])
    prob = svm_problem(y_train, x_train)
    m = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_val, x_val, m)
    E_CV += calError(p_label, y_val, x_val)
E_CV /= fold
print("E_CV: " + str(E_CV))

# prob = svm_problem(y_all, x_all)
# m = svm_train(prob, param)
# # svm_save_model("model.txt", m)
# # m = svm_load_model("model.txt")
# p_label, p_acc, p_val = svm_predict(y_all, x_all, m)
# # print(p_acc)
# err = calError(p_label, y_all, x_all)
# print(err)
# # print(p_val)