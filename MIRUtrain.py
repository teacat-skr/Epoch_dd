# INありなしでグラフの出力
import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd

fontsize = 25
x = range(4001)
y = []
csv_path = '/workspace/Epoch_dd/csv/IN_resnet18_cifar10_4000epochs_train.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
y.append(data[0][1])
co = 0
baf = 0.0
for i in range(len(data) - 1):
    co += 1
    baf += data[i + 1][1]
    if co == 391:
        y.append(baf / 391.0)
        co = 0
        baf = 0

y2 = []
csv_path = '/workspace/Epoch_dd/csv/SR_resnet18_cifar10_4000epochs_train.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
y2.append(data[0][1])
co = 0
baf = 0.0
for i in range(len(data) - 1):
    co += 1
    baf += data[i + 1][1]
    if co == 391:
        y2.append(baf / 391.0)
        co = 0
        baf = 0

y3 = []
csv_path = '/workspace/research/csv/resnet18_64_cifar10_train.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
y3.append(data[0][1])
co = 0
baf = 0.0
for i in range(len(data) - 1):
    co += 1
    baf += data[i + 1][1]
    if co == 391:
        y3.append(baf / 391.0)
        co = 0
        baf = 0
print(len(y3))
plt.rcParams["font.size"] = fontsize
plt.figure(figsize=(9,7))
# plt.title(f"Epoch-wise Double Descent w/ w/o PreTrain")
# plt.ylim(0.0, 0.5)
plt.text(x=1, y=0.1, s="(a)", fontsize=40)
plt.xlabel("Epoch", fontsize=fontsize)
plt.ylabel("Train Error", fontsize=fontsize)
plt.plot(x[1:], y3[1:], label="ResNet18[Nakkiran]")
plt.plot(x[1:], y[1:], label="ResNet18 w/ PT")
plt.plot(x[1:], y2[1:], label="ResNet18 w/o PT")
plt.legend(loc='upper right')
plt.xscale("log")
plt.tight_layout()
plt.savefig("MIRU_COMPtrain.png")
plt.savefig("MIRU_COMPtrain.pdf")