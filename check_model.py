#パラメータ確認用
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms;
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import math
from model import resnet18k
from model import mobilenetv2
import torchvision.models as models
# from transformers import ViTForImageClassification

import argparse
import random
import matplotlib.pyplot as plt
import csv 
import warnings
import os
import sys
import wandb
import numpy as np
import cv2
from PIL import Image

model = models.mobilenet_v2()
# model.classifier[1] = nn.Linear(in_features=in_features, out_features=10, bias=True)
# model.load_state_dict(torch.load(f"/workspace/Epoch_dd/model_weight1/IN_mobilenet_v2_cifar10_ln20pc_seed42/IN_mobilenet_v2_epoch0001_cifar10_ln20pc_seed42.pth"))
print(model.features)