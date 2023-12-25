import os
import csv
import matplotlib.pyplot as plt
import math
import pandas as pd

x = []
y = []
csv_path = '/workspace/Epoch_dd/csv/Model_wise_ST/SR_resnet18k-128_cifar10_4000epochs_test.csv'
data = pd.read_csv(csv_path)
data = data.values.tolist()
for i in range(len(data)):
    x.append(i + 1)
    y.append(data[i][2])

# plt.title(f"Epoch-wise Double Descent\n{os.path.basename(csv_path)[:-4]}")
plt.xscale("log")
plt.ylim(0.0, 0.5)
plt.xlabel("epoch")
plt.ylabel("Test Error")
plt.plot(x, y)
plt.savefig("EDD_" + os.path.basename(csv_path)[:-4] + ".png")
plt.savefig("EDD_" + os.path.basename(csv_path)[:-4] + ".pdf")