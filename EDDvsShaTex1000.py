import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np

from matplotlib.font_manager import FontProperties
# font_path = "/usr/share/fonts/truetype/migmix/migmix-1p-regular.ttf"
# font_prop = FontProperties(fname=font_path)
# plt.rcParams["font.family"] = font_prop.get_name()

fontsize = 25
epoch = 1000
maxepoch = 1000
seed = 42
model = "resnet18_frozen_fc"

x = []
y = []
y2 = []
csv_path = f'./csv/IN_{model}_cifar10_{epoch}epochs_ln20pc_seed42_test.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
minnum = 10000
maxshape = 0.0
mintex = 100.0
ishape = 1
itexture = 1
ind = -1
for i in range(len(data)):
    x.append(data[i][0])
    y.append(data[i][1] * 100)
    y2.append(data[i][2] * 10)
    if minnum > data[i][1]:
        minnum = data[i][1]
        ind = data[i][0]
    if(data[i][0] == maxepoch):
        break

# y3 = []
# csv_path = f'./csv/IN_{model}_cifar10_{epoch}epochs_ln20pc_seed42_edgetest.csv'
# data = pd.read_csv(csv_path)
# data = data.values.tolist()
# minnum = 10000
# maxshape = 0.0
# mintex = 100.0
# ishape = 1
# itexture = 1
# ind = -1
# for i in range(len(data)):
#     # x.append(data[i][0])
#     y3.append(data[i][1] * 100)
#     # y2.append(data[i][2] * 10)
#     if minnum > data[i][1]:
#         minnum = data[i][1]
#         ind = data[i][0]
#     if(data[i][0] == maxepoch):
#         break

train_x = []
train_y = []
csv_path = f'./csv/IN_{model}_cifar10_{epoch}epochs_ln20pc_seed42_train.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
count = 0
sub = 0.0
for i in range(len(data)):
    if(data[i][0] == 0):
        train_x.append(data[i][0])
        train_y.append(data[i][1] * 100)
    else:
        count += 1
        if(count == len(data) // epoch):
            train_x.append(data[i][0])
            train_y.append(sub / (len(data) // epoch))
            count = 0
            sub = 0
        else:
            sub += data[i][1] * 100
    if(data[i][0] == maxepoch):
        break

x2 =[]
Shapes = []
Textures = []
csv_path = f'./shape_texture_bias_csv/IN_resnet18_frozen_fc_cifar10_ln20pc_seed42.csv'
data = pd.read_csv(csv_path, header=None)
data = data.values.tolist()
for i in range(int(len(data) / 2)):
    x2.append(i + 1)
    Shapes.append(data[i * 2 + 1][0])
    Textures.append(data[i * 2 + 1][1])
    if maxshape < data[i * 2 + 1][0] and i > 5:
        maxshape = data[i * 2 + 1][0]
        ishape = i + 1
    if mintex > data[i * 2 + 1][1] and i > 5:
        mintex = data[i * 2 + 1][1]
        itexture = i + 1

fig, ax1 = plt.subplots()
plt.xticks(fontsize =fontsize)
plt.yticks(fontsize =fontsize)
ax2 = ax1.twinx()
ax2.tick_params(labelsize =fontsize)
fig.set_size_inches(12, 12)
plt.rcParams["font.size"] = fontsize
plt.xlabel("Epoch", fontsize=fontsize)
ax1.set_ylim(0.0, 80.0)
ax2.set_ylim(0.0, 80.0)
# plt.text(x=12 -2.5, y=Textures[-1] + 5, s="(a)", fontsize=fontsize)
# plt.text(x=63 -15, y=Textures[-1] + 5, s="(b)", fontsize=fontsize)
# plt.text(x=4000 -900, y=Textures[-1] + 5, s="(c)", fontsize=fontsize)
# plt.axvline(x=12, ymin=0.12, ymax=0.43 * 1.2, c="black")
# # plt.axvline(x=43, ymin=0.12, ymax=0.43 * 1.2, c="black")
# plt.axvline(x=63, ymin=0.12, ymax=0.43 * 1.2, c="black")
# plt.axvline(x=4000, ymin=0.12, ymax=0.43 * 1.2, c="black")

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
ax2.plot(x[1:], y[1:], label="TestError", linewidth = 2)
# ax2.plot(x[1:], y3[1:], label="EdgeTestError", linewidth = 2)
ax2.plot(train_x[1:], train_y[1:], label="TrainError", color="gray", linestyle="dashdot", linewidth = 2)
# plt.plot(x, y2, label="TestLoss")
ax1.set_ylabel("Shape/Texture Bias(%)", fontsize=fontsize + 10)
ax2.set_ylabel("Test/Train Error (%)", fontsize=fontsize + 10)
ax1.set_xlabel('Epoch', fontsize=fontsize + 10)
ax1.plot(x2, Shapes, label="ShapeBias", color="tab:red", linewidth = 2)
ax1.plot(x2, Textures,label="TextureBias", color="tab:green", linewidth = 2)
plt.xscale("log")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc='upper right', borderpad=1, frameon=False)
save_name = f"EDDvsShaTex{maxepoch}"
plt.savefig(save_name + ".png")
plt.savefig(save_name + ".pdf")

print("minimum TE epoch:%d", ind)
print("maximum Shape epoch:%d value:%d" %(ishape, maxshape))
print("minimum Texture epoch:%d value:%d"%(itexture, mintex))
    
