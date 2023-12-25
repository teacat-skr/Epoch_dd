# INありなしでグラフの出力
import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd

baf = "test"
fontsize = 25

x = []
y = []
csv_path = '/workspace/Epoch_dd/csv/IN_resnet18_cifar10_4000epochs_' + baf + '.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1])

y2 = []
csv_path = '/workspace/Epoch_dd/csv/SR_resnet18_cifar10_4000epochs_' + baf + '.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    y2.append(data[i][1])

y3 = []
csv_path = '/workspace/research/csv/resnet18_64_cifar10_' + baf + '.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    y3.append(data[i][1])
print(len(y3))
plt.rcParams["font.size"] = fontsize
plt.figure(figsize=(9,7))
# plt.title(f"Epoch-wise Double Descent w/ w/o PreTrain")
plt.ylim(0.0, 0.7)
plt.text(x=1, y=0.1, s="(b)", fontsize=40)
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Test Error", fontsize=fontsize)
plt.plot(x[1:], y3[1:], label="ResNet18[Nakkiran]")
plt.plot(x[1:], y[1:], label="ResNet18 w/ PT")
plt.plot(x[1:], y2[1:], label="ResNet18 w/o PT")
plt.legend(loc='upper right')
plt.xscale("log")
plt.tight_layout()
plt.savefig("MIRU_COMPtest.png")
plt.savefig("MIRU_COMPtest.pdf")