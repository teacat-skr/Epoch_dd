import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd

x = []
y = []
data = pd.read_csv('./csv/resnet18*128-cifar10-test.csv')
data = data.values.tolist()
for i in range(len(data)):
    x.append(i + 1)
    y.append(data[i][1])

plt.title("Epoch-wise Double Descent")
plt.xscale("log")
plt.ylim(0.0, 0.5)
plt.xlabel("ResNet18 width parameter")
plt.ylabel("Test Error")
plt.plot(x, y)
plt.savefig("./Epoch-wise_DoubleDescent.png")