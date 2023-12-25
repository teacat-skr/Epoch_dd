#固まっているか確認用
import os
import torch
import torch.nn as nn
import torchvision.models as models

i = 1
weight_path = f"/workspace/Epoch_dd/model_weight1/IN_resnet18_frozen_fc_cifar10_ln20pc_seed42/IN_resnet18_frozen_fc_epoch{i:04}_cifar10_ln20pc_seed42.pth"
# weight_path1 = f"/workspace/Epoch_dd/model_weight1/IN_resnet18_frozen_fc_cifar10_ln20pc_seed42/IN_resnet18_frozen_fc_epoch{1000:04}_cifar10_ln20pc_seed42.pth"
# # 重み読み込み
print("model weight path: {}".format(weight_path))
assert os.path.exists(weight_path)
state_dict = torch.load(weight_path)
print(state_dict["fc.weight"])
# state_dict1 = torch.load(weight_path1)
# print(state_dict["layer4.1.conv1.weight"] - state_dict1["layer4.1.conv1.weight"])

#残差接続削除確認用コード
model = models.resnet18(num_classes=10)
model.load_state_dict(state_dict)

# # モデルの概要を表示
# for i in model:
#     for j in i:
#         print(j)

