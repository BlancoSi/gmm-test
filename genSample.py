import numpy as np
import matplotlib.pyplot as plt

cov1 = np.mat("0.3 0;0 0.1")
cov2 = np.mat("0.2 0;0 0.3")
cov3 = np.mat("0.1 0;0 0.2")
mu1 = np.array([0, 1])
mu2 = np.array([2, 1])
mu3 = np.array([1, 1])

sample = np.zeros((150, 2))
sample[:30, :] = np.random.multivariate_normal(mean=mu1, cov=cov1, size=30)
sample[30:100, :] = np.random.multivariate_normal(mean=mu2, cov=cov2, size=70)
sample[100:, :] = np.random.multivariate_normal(mean=mu3, cov=cov3, size=50)
np.savetxt("sample.data", sample)

plt.plot(sample[:30, 0], sample[:30, 1], "bo")
plt.plot(sample[30:100, 0], sample[30:100, 1], "rs")
plt.plot(sample[100:, 0], sample[100:, 1], "go")
plt.show()
