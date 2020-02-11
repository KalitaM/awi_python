import numpy as np 
from numpy.linalg import norm
from utils import *
import matplotlib.pyplot as plt


data = np.load("compare-csref00350-iter00020fwd1.ttr_filter_output.csv.npy")
_, data_r = read_ttr_residual("res-csref00350-iter00020fwd1.ttr")


f = []
for i in range(data.shape[0]):
    d = data[i, :]
    f.append(d.T @ d)


f_r = []
for i in range(data_r.shape[0]):
    d = data_r[i, :]
    f_r.append(d.T @ d)


plt.plot(f / max(f), label="T * v")
plt.plot(f_r / max(f_r), label="residual")
plt.legend()
plt.show()