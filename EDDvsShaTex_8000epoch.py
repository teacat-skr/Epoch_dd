import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

x = []
y = []
y2 = []
csv_path = './csv/IN_resnet18_cifar10_8000epochs_test.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    x.append(i + 1)
    y.append(data[i][1] * 100)
    # y2.append(data[i][2] * 10)

x2 =[]
Shapes = []
Textures = []
csv_path = './8000epoch_1_1000_1slice.csv'
data = pd.read_csv(csv_path, header=None)
data = data.values.tolist()
for i in range(int(len(data) / 2)):
    x2.append(i + 1)
    Shapes.append(data[i * 2 + 1][0])
    Textures.append(data[i * 2 + 1][1])

csv_path = './8000epoch_1005_8000_5slice.csv'
data = pd.read_csv(csv_path, header=None)
data = data.values.tolist()
for i in range(int(len(data) / 2)):
    x2.append(i * 5 + 1005)
    Shapes.append(data[i * 2 + 1][0])
    Textures.append(data[i * 2 + 1][1])

def fix(window, a):
    #端点補正用関数
    for i in range((window - 1) // 2):
        a[i] *= window
        a[i] /= (i + 1 + window // 2)
    for i in range((window - 1) // 2):
        a[-i - 1] *= window 
        a[-i - 1] /= (i + 1 + window // 2)

window = 5 # 移動平均の範囲
w = np.ones(window)/window
Shapes = np.convolve(Shapes, w, mode='same')
fix(window, Shapes)
Textures = np.convolve(Textures, w, mode='same')
fix(window, Textures)

fig, ax1 = plt.subplots()
fig.set_size_inches(6, 6)
plt.xlabel("Epoch", fontsize=13)
plt.ylim(0.0, 100.0)
plt.plot(x, y, label="TestError")
# plt.plot(x, y2, label="TestLoss")
plt.ylabel("Test Error/Shape/Texture (%)", fontsize=13)
plt.plot(x2, Shapes, label="Shape")
plt.plot(x2, Textures,label="Texture")
plt.xscale("log")
plt.legend(loc='upper right')
save_name = "EDDvsShaTex_8000epoch"
plt.savefig(save_name + ".png")
plt.savefig(save_name + ".pdf")


    
