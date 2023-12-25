#transformを間違えていたので再計算用
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
import dd

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True, num_workers=2)

criterion = nn.CrossEntropyLoss().to(device)


for i in [12, 62, 3999]:
    baf =  f'{i + 1:04}'
    model = models.resnet18(pretrained=False, num_classes=10)
    load_weights = torch.load('/workspace/Epoch_dd/model_weight/IN_res18_4000ep/IN_resnet18_epoch' + baf + 'cifar10.pth')
    model.load_state_dict(load_weights)
    model.to(device)
    model.eval()
    test_acc, test_loss = dd.test(model, device, test_loader, criterion)
    with open('/workspace/Epoch_dd/csv/IN_resnet18_cifar10_4000epochs_test_aa.csv','a') as file:
        file.write(f"{i + 1},{1.0 - test_acc},{test_loss}" + "\n")