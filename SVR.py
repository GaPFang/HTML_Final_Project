# SVR with two constraints: non-negative and smaller than tot

from libsvm.svmutil import *
import numpy as np
import os
import sys
import random
import matplotlib.pyplot as plt
import json

total_day = 23 + 16
total_time = 72
total_station = 1322
N = total_day * total_day * total_station

repeat = 1

y_all, x_all = svm_read_problem("train.txt")

C = sys.argv[2]
gamma = sys.argv[3]
epsilon = sys.argv[4]

print("C = " + str(C) + ", gamma = " + str(gamma) + ", epsilon = " + str(epsilon), file=sys.stderr)

# E_CV = np.zeros((5, 5, 5)).tolist()
# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             param = svm_parameter('-s 3 -t 2 -c ' + str(C[i]) + ' -g ' + str(gamma[j]) + ' -p ' + str(epsilon[k]) + ' -q')
#             for r in range(repeat):
#                 mess = np.random.permutation(N).tolist()
#                 fold = [mess[:int(N/5)]] + [mess[int(N/5):int(2 * N/5)]] + [mess[int(2 * N/5):int(3 * N/5)]] + [mess[int(3 * N/5):int(4 * N/5)]] + [mess[int(4 * N/5):N]]
#                 for v in range(5):
#                     x_train = []
#                     y_train = []
#                     x_val = []
#                     y_val = []
#                     for j in range(5):
#                         if j == v:
#                             for k in range(int(N/5)):
#                                 x_val.append(x_all[fold[j][k]])
#                                 y_val.append(y_all[fold[j][k]])
#                         else:
#                             for k in range(int(N/5)):
#                                 x_train.append(x_all[fold[j][k]])
#                                 y_train.append(y_all[fold[j][k]])
#                     prob = svm_problem(y_train, x_train)
#                     m = svm_train(prob, param)
#                     p_label, p_acc, p_val = svm_predict(y_val, x_val, m)
#                     E_CV[i][j][k] += ((1 - p_acc[0]) / 100)
#             E_CV[i][j][k] /= (5 * repeat)
#             print("C = " + str(C[i]) + ", gamma = " + str(gamma[j]) + ", epsilon = " + str(epsilon[k]) + ", E_CV = " + str(E_CV[i][j][k]), file=sys.stderr)

        