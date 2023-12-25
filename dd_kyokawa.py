"""
# 2021/12/3作成: 
- torchvisionのmodels読み込み部分作成
    - argsから呼び出す
- csvを逐次書き出すに変更
- 
"""
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
import torchvision.models as models


import argparse
import random
import matplotlib.pyplot as plt
import csv 
import warnings

def get_model(args):
    """モデルの読み込み

    Arguments:
        args: 
    Returns:
        model
    """
    if args.model == "resnet18k":
        assert args.model_width is not None, "please check k value"
        model = resnet18k.make_resnet18k(k=args.model_width, num_classes=args.num_classes)
        args.model_fullname = "SR_resnet18k-{}".format(args.model_width)
    elif args.model == "resnet18":
        if args.pretrained == False:
            # model
            model = models.resnet18(pretrained=False, num_classes=args.num_classes)
            args.model_fullname = "SR_resnet18" # SR means scratch
        else: ## 事前学習モデルを利用する場合，一度重みを読み込んで，全結合層だけ初期化する
            model = models.resnet18(pretrained=True)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features=in_features, out_features=args.num_classes, bias=True)
            args.model_fullname = "IN_resnet18"
    return model
    
def sub():
    args = parse_args()
    #epoch数指定
    epoch = args.epoch
    label_noise_rate = args.label_noise_rate
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    #インスタンス変数にアクセスしてラベルの張替え
    for i in range(len(train_set.targets)):
        if(random.randint(0, 9999) < int(label_noise_rate * 10000)):
            train_set.targets[i] += random.randint(1, 9)
            train_set.targets[i] %= 10
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_train)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=True, num_workers=2)
    class_names = ('plane', 'car', 'bird', 'cat', 'dog', 'frog', 'ship', 'truck')
    
    model = get_model(args)
    model = model.to(device)
    print(args.model_fullname)

    # if device == 'cuda':
        # model = torch.nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    x1 = []
    x2 = range(epoch + 1)
    x1.append(0)

    #----------#
    ### csv 作成
    #----------#
    # 初期値取得
    train_acc, train_loss = test(model, device, test_loader, criterion) # 初期値取得
    test_acc, test_loss = test(model, device, test_loader, criterion)
    # train
    with open('./csv/{}-cifar10-train_epoch-{}.csv'.format(args.model_fullname, args.epoch),'w') as file:
        file.write("epoch,error,loss\n")
        file.write(f"0,{1.0 - train_acc},{train_loss}" + "\n")
    # test
    with open('./csv/{}-cifar10-test_epoch-{}.csv'.format(args.model_fullname, args.epoch),'w') as file:
        file.write("epoch,error,loss\n")
        file.write(f"0,{1.0 - test_acc},{test_loss}" + "\n")
    
    ## ----------------#
    ## training #
    # ----------------#
    for epoch in range(epoch):
        # Train and test a model.
        model.train()
        #trainの各数値はbatchごとに出す
        #calc_scoreの返却値のlossはbatch数で割っているので無視,loss.item()を用いる
        for batich_idx, (inputs, targets) in enumerate(train_loader):
            output_list = []
            target_list = []
            running_loss = 0.0
            xpoint = 0.0 + epoch + (float(batich_idx + 1) / len(train_loader))
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            output_list += [int(o.argmax()) for o in outputs]
            target_list += [int(t) for t in targets]
            running_loss += loss.item()

            train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)
            x1.append(xpoint)
            ## csv writer
            with open('./csv/{}-cifar10-train_epoch-{}.csv'.format(args.model_fullname, args.epoch),'a') as file:
                file.write(f"{xpoint},{1.0 - train_acc},{loss.item()}" + "\n")
            
            # if batich_idx % 100 == 0 and batich_idx != 0:
            #     stdout_temp = 'batch: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
            #     print(stdout_temp.format(batich_idx, len(train_loader), train_acc, loss.item()))

        ## これないとダメmodel.eval
        model.eval()
        test_acc, test_loss = test(model, device, test_loader, criterion)
        # csv writer
        with open('./csv/{}-cifar10-test_epoch-{}.csv'.format(args.model_fullname, args.epoch),'a') as file:
                file.write(f"{epoch + 1},{1.0 - test_acc},{test_loss}" + "\n")

        # Output score.
        stdout_temp = 'epoch: {:>3}, train acc: {:<8}, train loss: {:<8}, test acc: {:<8}, test loss: {:<8}'
        print(stdout_temp.format(epoch+1, train_acc, loss.item(), test_acc, test_loss))

        ## model save###
        # torch.save(model.state_dict(), './model_weight/resnet18*' + str(args.model_width) + '-cifar10-train.csv')
        # pytorchの慣例でpthファイルで保存する
        #実験のためにepochごとに保存
        model_path = f'./model_weight/{args.model_fullname}_epoch{(epoch + 1):04}cifar10.pth'
        torch.save(model.state_dict(), model_path)
    
    #errorのグラフ化
    # plt.title(args.model_fullname + " trained by Cifar10")
    # plt.xlim(0, epoch * 1.2)
    # plt.ylim(0, 1)
    # plt.xlabel("Epoch")
    # plt.ylabel("Erorr")
    # plt.plot(x1, trerr, label='train', linewidth=0.5)
    # plt.plot(x2, teerr, label='test')
    # plt.legend(loc='upper right')
    # plt.savefig("./output/{}TrainedByCifar10.png".format(args.model_fullname))
    # plt.savefig("./output/{}TrainedByCifar10.pdf".format(args.model_fullname))

    # plt.close()

    # #lossのグラフ化
    # plt.title(args.model_fullname + " trained by Cifar10")
    # plt.xlim(0, epoch * 1.2)
    # plt.ylim(0, max(max(trloss) * 1.2, max(teloss) * 1.2))
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.plot(x1, trloss, label='train', linewidth=0.5)
    # plt.plot(x2, teloss, label='test')
    # plt.legend(loc='upper right')
    # plt.savefig("./output/loss-{}TrainedByCifar10.png".format(args.model_fullname))
    # plt.savefig("./output/loss-{}TrainedByCifar10.pdf".format(args.model_fullname))
    

def train (model, device, train_loader, criterion, optimizer):
    model.train()
    output_list = []
    target_list = []
    running_loss = 0.0
    for batich_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_list += [int(o.argmax()) for o in outputs]
        target_list += [int(t) for t in targets]
        running_loss += loss.item()

        train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)
        if batich_idx % 100 == 0 and batich_idx != 0:
            stdout_temp = 'batch: {:>3}/{:<3}, train acc:{:<8}, train loss: {:<8}'
            print(stdout_temp.format(batich_idx, len(train_loader), train_acc, train_loss))
    train_acc, train_loss = calc_score(target_list, output_list, running_loss, train_loader)


    return train_acc, train_loss

def test(model, device, test_loader, criterion):
	model.eval()

	output_list = []
	target_list = []
	running_loss = 0.0
	with torch.no_grad():
		for batch_idx, (inputs, targets) in enumerate(test_loader):
			# Forward processing.
			inputs, targets = inputs.to(device), targets.to(device)
			outputs = model(inputs)
			loss = criterion(outputs, targets)
			
			# Set data to calculate score.
			output_list += [int(o.argmax()) for o in outputs]
			target_list += [int(t) for t in targets]
			running_loss += loss.item()
		
	test_acc, test_loss = calc_score(target_list, output_list, running_loss, test_loader)

	return test_acc, test_loss
def calc_score(true_list, predict_list, running_loss, data_loader):
    # import pdb;pdb.set_trace()
    # result = classification_report(true_list, predict_list, output_dict=True)
    # acc = round(result['accuracy'], 6)
    acc = accuracy_score(true_list, predict_list)
    loss = round(running_loss / len(data_loader), 6)

    return acc, loss

def parse_args():
    arg_parser = argparse.ArgumentParser(description="ResNet trained by CIFAR-10")
    
    arg_parser.add_argument("-k", "--model_width", type=int, default=1)
    arg_parser.add_argument("-e", "--epoch", type=int, default=4000)
    arg_parser.add_argument("--label_noise_rate", type=float, default=0.0)

    # 追加
    arg_parser.add_argument("--model", type=str, choices=["resnet18k", "resnet18"], help="モデルアーキテクチャの選択")
    arg_parser.add_argument("-pt", "--pretrained", action='store_true', help="事前学習モデルの利用．ResNet18のときのみ有効．引数をつけるとTrue")
    arg_parser.add_argument("--num_classes", type=int, default=10, help="分類クラス数")

    return arg_parser.parse_args()

if __name__ =='__main__':
    warnings.filterwarnings('ignore')
    import time
    start = time.perf_counter()
    sub()
    
    print(time.perf_counter() - start)
