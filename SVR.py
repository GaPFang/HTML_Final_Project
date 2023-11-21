# SVR with two constraints: non-negative and smaller than tot

from libsvm.svmutil import *
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
from collections import OrderedDict

def calError(y, y_hat):
    return np.sum(np.abs(y - y_hat)) / len(y)

fold = 1000
total_day = 23 + 16
total_time = 72
total_station = 1322
N = 3704832

repeat = 10

y_all, x_all = svm_read_problem("train.txt")
x_all = x_all[:N]
y_all = y_all[:N]

C = [10, 100, 1000]
gamma = [1, 10, 100]
epsilon = [0.01, 0.1, 1]

# print("C = " + str(C) + ", gamma = " + str(gamma) + ", epsilon = " + str(epsilon))

proFd = open("progress.txt", "w")
read_pipe = np.zeros((len(C), len(gamma), len(epsilon))).tolist()
mess = np.random.permutation(N).tolist()
folds = []
read_pipe = np.zeros((len(C), len(gamma), len(epsilon))).tolist()
for v in range(fold):
    folds.append(mess[int(v * N / fold):int((v + 1) * N / fold)])
for i in range(len(C)):
    for j in range(len(gamma)):
        for k in range(len(epsilon)):
            param = svm_parameter('-s 3 -t 2 -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k]) + ' -q')
            r, w = os.pipe()
            pid = os.fork()
            if pid == 0:
                E_CV = 0
                os.close(r)
                for r in range(repeat):
                    np.random.seed(np.random.seed(r))
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
                        # p_label, p_acc, p_val = svm_predict(y_val, x_val, m)
                        # E_CV += v
                        # E_CV += p_acc[1]
                        proFd.write("i, j, k, v = " + str(i) + ", " + str(j) + ", " + str(k) + ", " + str(v) + "\n")
                        proFd.flush()
                proFd.close()
                E_CV /= (repeat * fold)
                os.write(w, str(E_CV).encode('utf-8'))
                os._exit(0)
            os.close(w)
            read_pipe[i][j][k] = r

result = {}
for i in range(len(C)):
    for j in range(len(gamma)):
        for k in range(len(epsilon)):
            E_CV = float(os.read(read_pipe[i][j][k], 100).decode('utf-8'))
            result.update({E_CV: [C[i], gamma[j], epsilon[k]]})
result = OrderedDict(sorted(result.items()))
max_key_length = max(len(str(key)) for key in result)

f = open("result.txt", "w")
for i in result:
    f.write("E_CV: " + str(i).ljust(max_key_length) + " C: " + str(result[i][0]) + " gamma: " + str(result[i][1]) + " epsilon: " + str(result[i][2]) + "\n")
f.close()