# INありなしでグラフの出力
import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd

x = []
y = []
csv_path = '/workspace/Epoch_dd/csv/IN_resnet18_cifar10_4000epochs_test.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    x.append(i + 1)
    y.append(data[i][1])

y2 = []
csv_path = '/workspace/Epoch_dd/csv/SR_resnet18_cifar10_4000epochs_test.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    y2.append(data[i][1])

# plt.title(f"Epoch-wise Double Descent w/ w/o PreTrain")
plt.xscale("log")
plt.ylim(0.0, 0.5)
plt.xlabel("epoch")
plt.ylabel("Test Error")
plt.plot(x, y, label="w/ PT")
plt.plot(x, y2, label="w/o PT")
plt.legend(loc='upper right')
plt.savefig("SRvsIN_ResNet18_cifar10_EDD" + ".png")
plt.savefig("SRvsIN_ResNet18_cifar10_EDD" + ".pdf")