import argparse
import matplotlib.pyplot as plt
from gmm import *

DEBUG = True

options = argparse.ArgumentParser()
options.add_argument("-f", "--file", type=str, required=False, default='dot.txt', help='载入txt文件名称（默认为dot.txt）')
options.add_argument("-n", "--number", type=int, required=False, default=3, help='聚类的类别个数（默认为3组最大为7）')
options.add_argument("-t", "--time", type=int, required=False, default=100, help='迭代次数（默认为100次）')
args = options.parse_args()

Y = np.loadtxt(args.file)
matY = np.matrix(Y, copy=True)

K = args.number

mu, cov, alpha = GMM_EM(matY, K, args.time)

N = Y.shape[0]

gamma = getExpectation(matY, mu, cov, alpha)

category = gamma.argmax(axis=1).flatten().tolist()[0]

myclass = []
mycolar = ['rs', 'ko', 'gv', 'c2', 'm3', 'yh', 'bd']
mylabel = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7']
for num in range(K):
    myclass.append(np.array([Y[i] for i in range(N) if category[i] == num]))
for num in range(K):
    plt.plot(myclass[num][:, 0], myclass[num][:, 1], mycolar[num], label = mylabel[num])
plt.legend(loc="best")
plt.title("GMM-EM")
plt.show()
