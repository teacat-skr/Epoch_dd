#比較のために学習曲線と偏重度を分離して表示
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

fontsize = 30
epoch = 1000

def learning_curv(path_names, identify_name, color_list, graph_name):
    plt.figure(figsize=(11, 11))
    plt.xticks(fontsize =fontsize)
    plt.yticks(fontsize =fontsize)
    plt.rcParams["font.size"] = fontsize
    plt.xscale("log")
    plt.xlabel('Epoch', fontsize=fontsize + 20)
    plt.xticks(fontsize=fontsize+10)
    plt.yticks(fontsize=fontsize+10)
    plt.ylim(0.0, 100.0)
    plt.ylabel("Test/Train Error (%)", fontsize=fontsize + 20)

    for name, id, color in zip(path_names, identify_name, color_list):
        x = []
        y = []
        y2 = []
        csv_path = f'./csv/{name}_test.csv'
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
            if(data[i][0] == epoch):
                break

        train_x = []
        train_y = []
        csv_path = f'./csv/{name}_train.csv'
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
                if(count == 391):
                    train_x.append(data[i][0])
                    train_y.append(sub / 391.0)
                    count = 0
                    sub = 0
                else:
                    sub += data[i][1] * 100
            if(data[i][0] == epoch):
                break
        
        plt.plot(x[1:], y[1:], label=id  + ":" +  "TestError", linewidth = 2, color=color)
        plt.plot(train_x[1:], train_y[1:], label=id  + ":" +  "TrainError", linestyle="dashdot", linewidth = 2, color=color)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    vrious_num = len(handles) // 2
    handles[0:vrious_num], handles[vrious_num:] = handles[0::2], handles[1::2]
    labels = [""]*len(labels)
    labels[0:vrious_num] = identify_name
    plt.legend(handles, labels, ncol=2, title="Test    Train", markerfirst=False, markerscale=2, alignment="right", handlelength=3.2, columnspacing=0, fontsize=fontsize+5, title_fontsize=fontsize+10, labelspacing=0.2, frameon=False)
    plt.tight_layout()
    
    save_name = "learning_curv"
    plt.savefig("output/" + graph_name + "_" + save_name + ".png", bbox_inches='tight', pad_inches=0)
    plt.savefig("output/" + graph_name + "_" + save_name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()

def sha_tex(path_names, identify_name, color_list, graph_name):
    plt.figure(figsize=(11, 11))
    plt.rcParams["font.size"] = fontsize
    plt.xscale("log")
    plt.xticks(fontsize=fontsize+10)
    plt.yticks(fontsize=fontsize+10)
    plt.ylim(0.0, 80.0)
    plt.ylabel("Shape/Texture Bias(%)", fontsize=fontsize + 20)
    plt.xlabel('Epoch', fontsize=fontsize + 20)


    for name, id, color in zip(path_names, identify_name, color_list):
        x2 =[]
        Shapes = []
        Textures = []
        maxshape = 0.0
        mintex = 100.0
        ishape = 1
        itexture = 1
        ind = -1
        csv_path = f'./shape_texture_bias_csv/{name}.csv'
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

        # csv_path = f'./shape_texture_bias_csv/{name}1.csv'
        # data = pd.read_csv(csv_path, header=None)
        # data = data.values.tolist()
        # for i in range(int(len(data) / 2)):
        #     x2.append(i * 5 + 1005)
        #     Shapes.append(data[i * 2 + 1][0])
        #     Textures.append(data[i * 2 + 1][1])
        #     if maxshape < data[i * 2 + 1][0]:
        #         maxshape = data[i * 2 + 1][0]
        #         ishape = i * 5 + 1005
        #     if mintex > data[i * 2 + 1][1]:
        #         mintex = data[i * 2 + 1][1]
        #         itexture = i * 5 + 1005
        
        window = 5 # 移動平均の範囲
        w = np.ones(window)/window
        Shapes = np.convolve(Shapes, w, mode='same')
        fix(window, Shapes)
        Textures = np.convolve(Textures, w, mode='same')
        fix(window, Textures)
        
        plt.plot(x2, Textures,label=id + ":" + "TextureBias", linewidth = 2, color=color)
        plt.plot(x2, Shapes, label=id + ":" + "ShapeBias", linestyle="dashed", linewidth = 2, color=color)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    vrious_num = len(handles) // 2
    handles[0:vrious_num], handles[vrious_num:] = handles[0::2], handles[1::2]
    labels = [""]*len(labels)
    labels[0:vrious_num] = identify_name
    plt.legend(handles, labels, ncol=2, title="Texture Shape", markerfirst=False, markerscale=2, alignment="right", handlelength=3.4, columnspacing=0, fontsize=fontsize+5, title_fontsize=fontsize+10, labelspacing=0.2, frameon=False)
    plt.tight_layout()
    
    save_name = "sha_tex"
    plt.savefig("output/" + graph_name + "_" + save_name + ".png", bbox_inches='tight', pad_inches=0)
    plt.savefig("output/" + graph_name + "_" + save_name + ".pdf", bbox_inches='tight', pad_inches=0)
    plt.clf()



def fix(window, a):
    #端点補正用関数
    for i in range((window - 1) // 2):
        a[i] *= window
        a[i] /= (i + 1 + window // 2)
    for i in range((window - 1) // 2):
        a[-i - 1] *= window 
        a[-i - 1] /= (i + 1 + window // 2)



if __name__ =='__main__':

    color_list = ['#1f77b4', '#ff7f0e', '#2ca02c']

    #pre-train
    identify_name = ["ImageNet", "Scratch"]
    path_names = ["IN_resnet18_cifar10_4000epochs_ln20pc","SR_resnet18_cifar10_4000epochs_ln20pc"]
    learning_curv(path_names, identify_name, color_list, "INSR")
    path_names = ["IN_res18_4000ep_cifar10_ln20pc", "SR_res18_4000ep_cifar10_ln20pc"]
    sha_tex(path_names, identify_name, color_list, "INSR")

    #labelnoise
    identify_name = ["LN20%", "LN  0%"]
    path_names = ["IN_resnet18_cifar10_4000epochs_ln20pc","IN_resnet18_cifar10_4000epochs_ln0pc"]
    learning_curv(path_names, identify_name, color_list, "ln_ab")
    path_names = ["IN_res18_4000ep_cifar10_ln20pc", "IN_res18_4000ep_cifar10_ln0pc"]
    sha_tex(path_names, identify_name, color_list, "ln_ab")

    # dataset
    identify_name = ["CIFAR  10", "CIFAR100"]
    path_names = ["SR_resnet18_cifar10_4000epochs_ln20pc","SR_resnet18_cifar100_1000epochs_ln20pc_seed42"]
    learning_curv(path_names, identify_name, color_list, "SR_dataset")
    path_names = ["SR_res18_4000ep_cifar10_ln20pc", "SR_resnet18_cifar100_ln20pc_seed42"]
    sha_tex(path_names, identify_name, color_list, "SR_dataset")

    identify_name = ["CIFAR  10", "CIFAR100"]
    path_names = ["IN_resnet18_cifar10_4000epochs_ln20pc","IN_resnet18_cifar100_4000epochs_ln20pc"]
    learning_curv(path_names, identify_name, color_list, "IN_dataset")
    path_names = ["IN_res18_4000ep_cifar10_ln20pc", "IN_res18_4000ep_cifar100_ln20pc"]
    sha_tex(path_names, identify_name, color_list, "IN_dataset")


    # model
    identify_name = ["ResNet18", "ResNet50"]
    path_names = ["SR_resnet18_cifar10_4000epochs_ln20pc","SR_resnet50_cifar10_1000epochs_ln20pc_seed42"]
    learning_curv(path_names, identify_name, color_list, "SR_model")
    path_names = ["SR_res18_4000ep_cifar10_ln20pc", "SR_resnet50_cifar10_ln20pc_seed42"]
    sha_tex(path_names, identify_name, color_list, "SR_model")
    
    #cnn_model
    identify_name = ["DenseNet", "MobileNet", "EfficientNet"]
    path_names = ["IN_densenet121_cifar10_1000epochs_ln20pc_seed42","IN_mobilenetV2_cifar10_1000epochs_ln20pc_seed42","IN_efficientnetb0_cifar10_1000epochs_ln20pc_seed42"]
    learning_curv(path_names, identify_name, color_list, "cnn_model")
    path_names = ["IN_densenet121_cifar10_ln20pc_seed42","IN_mobilenet_v2_cifar10_ln20pc_seed42","IN_efficientnet_b0_cifar10_ln20pc_seed42"]
    sha_tex(path_names, identify_name, color_list, "cnn_model")
