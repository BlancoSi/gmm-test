
# -*- coding: utf-8 -*-

# ----------------------------------------------------

# Copyright (c) 2017, Wray Zheng. All Rights Reserved.

# Distributed under the BSD License.

# ----------------------------------------------------

import argparse
import matplotlib.pyplot as plt
from gmm import *

# 设置调试模式
DEBUG = True

# 命令行参数
options = argparse.ArgumentParser()
options.add_argument("-f", "--file", type=str, required=True, help='载入txt文件名称（必需，由xls另存为utf8编码txt文件）')
options.add_argument("-n", "--number", type=int, required=False, default=3, help='聚类的类别个数（默认为3组最大为7）')
options.add_argument("-t", "--time", type=int, required=False, default=100, help='迭代次数（默认为100次）')
args = options.parse_args()

# 载入数据集文件
Y = np.loadtxt(args.file)
matY = np.matrix(Y, copy=True)

# 载入聚类的类别个数，目前仅支持为3
K = args.number

# 载入迭代次数，计算 GMM 模型参数
mu, cov, alpha = GMM_EM(matY, K, args.time)

# 根据 GMM 模型，对样本数据进行聚类，一个模型对应一个类别
N = Y.shape[0]

# 求当前模型参数下，各模型对样本的响应度矩阵
gamma = getExpectation(matY, mu, cov, alpha)

# 对每个样本，求响应度最大的模型下标，作为其类别标识
category = gamma.argmax(axis=1).flatten().tolist()[0]

# 将每个样本放入对应类别的列表中
myclass = []
for num in range(K):
    myclass.append(np.array([Y[i] for i in range(N) if category[i] == num]))

# 绘制聚类结果
mycolar = ['rs', 'ko', 'gv', 'c2', 'm3', 'yh', 'bd']
mylabel = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']
for num in range(K):
    plt.plot(myclass[num][:, 0], myclass[num][:, 1], mycolar[num], label = mylabel[num])
plt.legend(loc="best")
plt.title("GMM-EM")
plt.show()
