#学習過程のモデルが出す認識結果の確度をcsv出力
#正解であるはずのクラスの確度をcsv化
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms;
import math
import torchvision.models as models
import csv 
from PIL import Image
import glob
import os

epoch = 1000
name = ""
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = models.resnet18(pretrained=False, num_classes=10)

l = [[0 for i in range(epoch + 2)] for j in range(10000)]
img_list = [[0 for i in range(2)] for j in range(10000)]
num = -1
count = 0
batch_list = []
for i in range(10):
    files = glob.glob("/workspace/cifar10_raw/default/{}/*".format(i))
    for f in files:
        img = Image.open(f)
        test = transforms.functional.to_tensor(img)
        test = transforms.functional.normalize(test, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        test_batch = test.to(device)
        baf = os.path.splitext(os.path.basename(f))[0]
        num = int(baf)
        img_list[num][0] = i
        img_list[num][1] = test_batch


print("load Image")
for i in range(10000):
    l[i][0] = i
    l[i][1] = img_list[i][0]
    batch_list.append(img_list[i][1])
new_batch = torch.stack(batch_list, 0)

for k in range(epoch):
    load_weights = torch.load(f'/workspace/Epoch_dd/model_weight/IN_res18_4000ep_cifar10_ln0pc/IN_resnet18_epoch{(k+1):04}_cifar10_ln0pc.pth')
    model.load_state_dict(load_weights)
    model.to(device)
    model.eval()
    m = nn.Softmax(dim=1)
    baf = m(model(new_batch)).tolist()
    #softmaxで確率に変換
    for i in range(10000):
        l[i][k + 2] = baf[i][l[i][1]]
    print(k)

with open('./comp_epoch_result_prob.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(l)
