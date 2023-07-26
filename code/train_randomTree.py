# encoding: utf-8
import argparse
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import openpyxl

from sklearn.metrics import mean_squared_error

import numpy as np
import torch
import torch.nn as nn
# 定义和训练神经网络的类和函数。它包括各种层、激活函数、损失函数和配置神经网络结构的实用工具。
import torch.optim as optim
# 这个模块提供了用于在训练过程中更新神经网络模型权重的优化算法。
import network  # 引入自定义文件network.py
import loss
import pre_process as prep
from Origin_PerformanceMeasure import Origin_PerformanceMeasure
# 这三个是自己定义的
import torch.utils.data as util_data  # To use 'DataLoader()'
import lr_schedule
from data_list import ImageList
from torch.autograd import Variable
# 貌似已经被弃用，主要是为了允许在安详传播的过程中进行自动微分来计算梯度
import math

optim_dict = {"SGD": optim.SGD}  #键值对设置

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import time
import random

import time
from PIL import Image
#
#
# def get_fea_lab(what, loader, model, gpu):
#     start_test = True
#     iter_what = iter(loader[what])
#     for i in range(len(loader[what])):
#         data = iter_what.next()
#         inputs = data[0]
#         labels = data[1]
#         if gpu:
#             inputs = Variable(inputs.cuda())
#             labels = Variable(labels.cuda())
#         else:
#             inputs = Variable(inputs)
#             labels = Variable(labels)
#
#         outputs = model(inputs)  # get features
#
#         if start_test:
#             all_output = outputs.data.float()
#             all_label = labels.data.float()
#             start_test = False
#         else:
#             all_output = torch.cat((all_output, outputs.data.float()), 0)
#             all_label = torch.cat((all_label, labels.data.float()), 0)
#
#     # all_label_list = all_label.view(-1.1).cpu().numpy()
#     # all_output_list = all_output.view(-1,1).cpu().numpy()
#     all_label_list = all_label.cpu().numpy()
#     all_output_list = all_output.cpu().numpy()
#     return all_output_list, all_label_list
#
#
# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     # random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
#
# def tsne(df_fea, df_lab):
#     # print(df_fea, df_lab)
#     tsne = TSNE(2, 38, 20, 600)
#     result = tsne.fit_transform(df_fea)
#     colors = ['c', 'r']
#     idx_1 = [i1 for i1 in range(len(df_lab)) if df_lab[i1] == 0]
#     fig1 = plt.scatter(result[idx_1, 0], result[idx_1, 1], 20, color=colors[0], label='Clean')
#
#     idx_2 = [i2 for i2 in range(len(df_lab)) if df_lab[i2] == 1]
#     fig2 = plt.scatter(result[idx_2, 0], result[idx_2, 1], 20, color=colors[1], label='Defective')
#
#
# def get_tsne_img(what, loader, model, gpu=True):
#     fea, lab = get_fea_lab(what, loader, model, gpu)
#     tsne(fea, lab)
#     plt.legend()
#     plt.xticks([])
#     plt.yticks([])
#
#     if what == 'train':
#         plt.savefig(
#             '../temp/tsne/' + args.source + '-' + args.target + '.pdf')
#     else:
#         plt.savefig(
#             '../temp/tsne/' + args.target + '-' + args.source + '.pdf')
#     plt.close()
#
#
#
#
#
# def image_classification_test(loader, loader1, test_10crop=True, gpu=True):
#     start_test = True
#     names = []
#     if test_10crop:
#         iter_test = [iter(loader['test' + str(i)]) for i in range(10)]
#         for i in range(len(loader['test0'])):
#             data = [iter_test[j].next() for j in range(10)]
#             inputs = [data[j][0] for j in range(10)]
#             names.append(data[0][2])
#             labels = data[0][1]
#             if gpu:
#                 for j in range(10):
#                     inputs[j] = inputs[j].numpy().flatten()  # Convert input images to 1D arrays
#                 labels = labels.numpy()
#             else:
#                 for j in range(10):
#                     inputs[j] = inputs[j].numpy().flatten()
#                 labels = labels.numpy()
#
#             inputs = np.concatenate(inputs, axis=1)  # Concatenate the 1D arrays
#             inputs = inputs.reshape(data[0].shape[0], -1)
#             rf_model = RandomForestRegressor()
#             rf_model.fit(inputs, labels)
#
#             # Make predictions using the trained random forest classifier
#             outputs = rf_model.predict(inputs)
#
#             if start_test:
#                 all_output = outputs.astype(float)
#                 all_label = labels.astype(float)
#                 start_test = False
#             else:
#                 all_output = np.concatenate((all_output, outputs.astype(float)), axis=0)
#                 all_label = np.concatenate((all_label, labels.astype(float)), axis=0)
#     else:
#         iter_test = iter(loader["test"])
#         iter_test1 = iter(loader1["test"])
#         data1 = next(iter_test1)
#         inputs1 = data1[0]
#         labels1 = data1[1]
#         inputs1 = inputs1.numpy().flatten()
#         labels1 = labels1.numpy()
#         inputs1 = inputs1.reshape(data1[0].shape[0], -1)
#         for _ in range(len(loader["test"])):
#             data = next(iter_test)
#             inputs = data[0]
#             labels = data[1]
#             names.append(data[2])
#             if gpu:
#                 inputs = inputs.numpy().flatten()
#                 labels = labels.numpy()
#             else:
#                 inputs = inputs.numpy().flatten()
#                 labels = labels.numpy()
#
#
#             inputs = inputs.reshape(data[0].shape[0], -1)
#             rf_model = RandomForestRegressor()
#             rf_model.fit(inputs, labels)
#             # Make predictions using the trained random forest classifier
#
#             outputs = rf_model.predict(inputs1)
#
#             if start_test:
#                 all_output = outputs.astype(float)
#                 all_label = labels1.astype(float)
#                 start_test = False
#             else:
#                 all_output = np.concatenate((all_output, outputs.astype(float)), axis=0)
#                 all_label = np.concatenate((all_label, labels1.astype(float)), axis=0)
#
#     mse = mean_squared_error(all_label, all_output)
#     return mse
#
#
# def transfer_classification(config):
#     # 定义一个字典类型变量
#     prep_dict = {}
#     # Add kry-value pairs for 'prep_dict'
#     # 数据预处理
#     for prep_config in config["prep"]:
#         prep_dict[prep_config["name"]] = {}
#         if prep_config["type"] == "image":
#             prep_dict[prep_config["name"]]["test_10crop"] = prep_config["test_10crop"]
#             # Call image_train() in pre_process.py
#             prep_dict[prep_config["name"]]["train"] = prep.image_train(resize_size=prep_config["resize_size"],
#                                                                        crop_size=prep_config["crop_size"])
#             #某个标准化操作
#             # call image_test()
#
#             if prep_config["test_10crop"]:
#                 prep_dict[prep_config["name"]]["test"] = prep.image_test_10crop(resize_size=prep_config["resize_size"],
#                                                                                 crop_size=prep_config["crop_size"])
#             else:
#                 prep_dict[prep_config["name"]]["test"] = prep.image_test(resize_size=prep_config["resize_size"],
#                                                                          crop_size=prep_config["crop_size"])
#
#     # class_criterion = nn.CrossEntropyLoss()         ##交叉熵损失函数
#     # class_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 2]))
#
#     class_criterion = nn.MSELoss()  ##mse损失函数
#     loss_config = config["loss"]
#     # 如果在配置文件的loss部分中，name属性指定为DAN，那么表示使用DAN（Domain Adversarial Neural Networks）方法来进行域适应（domain adaptation）学习。
#     # DAN是一种常用的域适应方法，它通过对抗训练的方式来使得特征提取器对源域和目标域的特征表示具有相同的分布，从而提高模型的泛化性能。在具体实现中，DAN使用一个域分类器来判
#     # 别输入的特征属于源域还是目标域，同时通过最小化这个分类器的损失来训练特征提取器，使得提取到的特征能够欺骗域分类器，使得域分类器无法判断输入的特征是来自源域还是目标域。
#     transfer_criterion = loss.loss_dict[loss_config["name"]]
#     if "params" not in loss_config:
#         loss_config["params"] = {}
#
#     ## prepare data
#     dsets = {}
#     dset_loaders = {}
#     for data_config in config["data"]:
#         dsets[data_config["name"]] = {}  # 创建字典元素
#         dset_loaders[data_config["name"]] = {}  # 创建一个字典元素
#         if data_config["type"] == "image":
#             # Creat a 'ImageList' object，读取txt文件中的数据赋值给字典元素'train'
#             dsets[data_config["name"]]["train"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
#                                                             transform=prep_dict[data_config["name"]]["train"])
#             # 创建一个DataLoader对象，为字典'dset_loaders'中嵌套字典元素'train'赋值
#             dset_loaders[data_config["name"]]["train"] = util_data.DataLoader(dsets[data_config["name"]]["train"],
#                                                                               batch_size=data_config["batch_size"][
#                                                                                   "train"], shuffle=True, num_workers=4)
#             if "test" in data_config["list_path"]:
#                 if prep_dict[data_config["name"]]["test_10crop"]:
#                     for i in range(10):  # 10?
#                         dsets[data_config["name"]]["test" + str(i)] = ImageList(
#                             open(data_config["list_path"]["test"]).readlines(),
#                             transform=prep_dict[data_config["name"]]["test"]["val" + str(i)]
#                         )
#                         dset_loaders[data_config["name"]]["test" + str(i)] = util_data.DataLoader(
#                             dsets[data_config["name"]]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
#                             shuffle=False, num_workers=4)
#                 else:
#                     dsets[data_config["name"]]["test"] = ImageList(open(data_config["list_path"]["test"]).readlines(),
#                                                                    transform=prep_dict[data_config["name"]]["test"])
#
#                     dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(dsets[data_config["name"]]["test"],
#                                                                                      batch_size=
#                                                                                      data_config["batch_size"]["test"],
#                                                                                      shuffle=False, num_workers=4)
#             else:
#                 if prep_dict[data_config["name"]]["test_10crop"]:
#                     for i in range(10):
#                         dsets[data_config["name"]]["test" + str(i)] = ImageList(
#                             open(data_config["list_path"]["train"]).readlines(),
#                             transform=prep_dict[data_config["name"]]["test"]["val" + str(i)])
#                         dset_loaders[data_config["name"]]["test" + str(i)] = util_data.DataLoader(
#                             dsets[data_config["name"]]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
#                             shuffle=False, num_workers=4)
#                 else:
#                     dsets[data_config["name"]]["test"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
#                                                                    transform=prep_dict[data_config["name"]]["test"])
#                     dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(dsets[data_config["name"]]["test"],
#                                                                                      batch_size=
#                                                                                      data_config["batch_size"]["test"],
#                                                                                      shuffle=False, num_workers=4)
#     # print(dsets[data_config["name"]]["train"])
#     # print(dsets[data_config["name"]]["test"])
#
#


def get_fea_lab(what, loader, model, gpu):
    start_test = True
    iter_what = iter(loader[what])
    for i in range(len(loader[what])):
        data = iter_what.next()
        inputs = data[0]
        labels = data[1]
        if gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs = Variable(inputs)
            labels = Variable(labels)

        outputs = model(inputs)  # get features

        if start_test:
            all_output = outputs.data.float()
            all_label = labels.data.float()
            start_test = False
        else:
            all_output = torch.cat((all_output, outputs.data.float()), 0)
            all_label = torch.cat((all_label, labels.data.float()), 0)

    # all_label_list = all_label.view(-1.1).cpu().numpy()
    # all_output_list = all_output.view(-1,1).cpu().numpy()
    all_label_list = all_label.cpu().numpy()
    all_output_list = all_output.cpu().numpy()
    return all_output_list, all_label_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tsne(df_fea, df_lab):
    # print(df_fea, df_lab)
    tsne = TSNE(2, 38, 20, 600)
    result = tsne.fit_transform(df_fea)
    colors = ['c', 'r']
    idx_1 = [i1 for i1 in range(len(df_lab)) if df_lab[i1] == 0]
    fig1 = plt.scatter(result[idx_1, 0], result[idx_1, 1], 20, color=colors[0], label='Clean')

    idx_2 = [i2 for i2 in range(len(df_lab)) if df_lab[i2] == 1]
    fig2 = plt.scatter(result[idx_2, 0], result[idx_2, 1], 20, color=colors[1], label='Defective')


def get_tsne_img(what, loader, model, gpu=True):
    fea, lab = get_fea_lab(what, loader, model, gpu)
    tsne(fea, lab)
    plt.legend()
    plt.xticks([])
    plt.yticks([])

    if what == 'train':
        plt.savefig(
            '../temp/tsne/' + args.source + '-' + args.target + '.pdf')
    else:
        plt.savefig(
            '../temp/tsne/' + args.target + '-' + args.source + '.pdf')
    plt.close()

def svr_test(loader,seed):
    test_data_all = []
    test_labels_all = []

    for data in loader["test"]:
        test_data_all.append(data[0])
        test_labels_all.append(data[1])

    X_train = torch.cat(test_data_all, dim=0)  #[116,3,224,224]
    X_train = X_train.reshape(X_train.shape[0],-1)
    y_train = torch.cat(test_labels_all, dim=0) #[116]


    rf_model = RandomForestRegressor(random_state=seed)
    rf_model.fit(X_train, y_train)

    return rf_model

def svr_predict(model,loader):

    test_data_all = []
    test_labels_all = []

    for data in loader["test"]:
        test_data_all.append(data[0])
        test_labels_all.append(data[1])

    X = torch.cat(test_data_all, dim=0)
    X = X.reshape(X.shape[0], -1)
    y = torch.cat(test_labels_all, dim=0)



    # 使用测试集来评估模型的性能
    y_pred = model.predict(X)

    p = Origin_PerformanceMeasure(y, y_pred)
    pofb = p.getPofb()
    print("pofb:", pofb)
    return pofb

def transfer_classification(config,classnum,seed):
    # 定义一个字典类型变量
    prep_dict = {}
    # Add kry-value pairs for 'prep_dict'
    # 数据预处理
    for prep_config in config["prep"]:
        prep_dict[prep_config["name"]] = {}
        if prep_config["type"] == "image":
            prep_dict[prep_config["name"]]["test_10crop"] = prep_config["test_10crop"]
            # Call image_train() in pre_process.py
            prep_dict[prep_config["name"]]["train"] = prep.image_train(resize_size=prep_config["resize_size"],
                                                                       crop_size=prep_config["crop_size"])
            # call image_test()

            if prep_config["test_10crop"]:
                prep_dict[prep_config["name"]]["test"] = prep.image_test_10crop(resize_size=prep_config["resize_size"],
                                                                                crop_size=prep_config["crop_size"])
            else:
                prep_dict[prep_config["name"]]["test"] = prep.image_test(resize_size=prep_config["resize_size"],
                                                                         crop_size=prep_config["crop_size"])

    # class_criterion = nn.CrossEntropyLoss()         ##交叉熵损失函数
    # class_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1, 1, 2]))

    class_criterion = nn.MSELoss()  ##mse损失函数
    loss_config = config["loss"]
    # 如果在配置文件的loss部分中，name属性指定为DAN，那么表示使用DAN（Domain Adversarial Neural Networks）方法来进行域适应（domain adaptation）学习。
    # DAN是一种常用的域适应方法，它通过对抗训练的方式来使得特征提取器对源域和目标域的特征表示具有相同的分布，从而提高模型的泛化性能。在具体实现中，DAN使用一个域分类器来判
    # 别输入的特征属于源域还是目标域，同时通过最小化这个分类器的损失来训练特征提取器，使得提取到的特征能够欺骗域分类器，使得域分类器无法判断输入的特征是来自源域还是目标域。
    transfer_criterion = loss.loss_dict[loss_config["name"]]
    if "params" not in loss_config:
        loss_config["params"] = {}

    ## prepare data
    dsets = {}
    dset_loaders = {}
    for data_config in config["data"]:
        dsets[data_config["name"]] = {}  # 创建字典元素
        dset_loaders[data_config["name"]] = {}  # 创建一个字典元素
        if data_config["type"] == "image":
            # Creat a 'ImageList' object，读取txt文件中的数据赋值给字典元素'train'
            dsets[data_config["name"]]["train"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
                                                            transform=prep_dict[data_config["name"]]["train"])
            # 创建一个DataLoader对象，为字典'dset_loaders'中嵌套字典元素'train'赋值
            dset_loaders[data_config["name"]]["train"] = util_data.DataLoader(dsets[data_config["name"]]["train"],
                                                                              batch_size=data_config["batch_size"][
                                                                                  "train"], shuffle=True, num_workers=4)
            if "test" in data_config["list_path"]:
                if prep_dict[data_config["name"]]["test_10crop"]:
                    for i in range(10):  # 10?
                        dsets[data_config["name"]]["test" + str(i)] = ImageList(
                            open(data_config["list_path"]["test"]).readlines(),
                            transform=prep_dict[data_config["name"]]["test"]["val" + str(i)]
                        )
                        dset_loaders[data_config["name"]]["test" + str(i)] = util_data.DataLoader(
                            dsets[data_config["name"]]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
                            shuffle=False, num_workers=4)
                else:
                    dsets[data_config["name"]]["test"] = ImageList(open(data_config["list_path"]["test"]).readlines(),
                                                                   transform=prep_dict[data_config["name"]]["test"])

                    dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(dsets[data_config["name"]]["test"],
                                                                                     batch_size=
                                                                                     data_config["batch_size"]["test"],
                                                                                     shuffle=False, num_workers=4)
            else:
                if prep_dict[data_config["name"]]["test_10crop"]:
                    for i in range(10):
                        dsets[data_config["name"]]["test" + str(i)] = ImageList(
                            open(data_config["list_path"]["train"]).readlines(),
                            transform=prep_dict[data_config["name"]]["test"]["val" + str(i)])
                        dset_loaders[data_config["name"]]["test" + str(i)] = util_data.DataLoader(
                            dsets[data_config["name"]]["test" + str(i)], batch_size=data_config["batch_size"]["test"],
                            shuffle=False, num_workers=4)
                else:
                    dsets[data_config["name"]]["test"] = ImageList(open(data_config["list_path"]["train"]).readlines(),
                                                                   transform=prep_dict[data_config["name"]]["test"])
                    dset_loaders[data_config["name"]]["test"] = util_data.DataLoader(dsets[data_config["name"]]["test"],
                                                                                     batch_size=
                                                                                     data_config["batch_size"]["test"],
                                                                                     shuffle=False, num_workers=4)

    class_num = 4# ??



    grid = svr_test(dset_loaders["source"],seed)
    result = svr_predict(grid,dset_loaders["target"])
    return result

def get_faults_num(path):
    # 打开文本文件，读取所有行
    with open(path, 'r') as f:
        lines = f.readlines()
    # 定义一个集合，用于存储所有出现过的数字
    nums_set = set()
    # 遍历每一行文本，对每一行进行处理
    for line in lines:
        # 将一行文本按照空格分割为两个部分，取出右边的数字部分
        num_str = line.split('\t')[-1].strip()

        # 如果该数字部分不为空，则将其转换为整数，并加入集合
        if num_str:
            nums_set.add(int(num_str))

    # 输出集合的长度，即为不同数字种类的数量
    # print(len(nums_set))
    return len(nums_set)


if __name__ == "__main__":
    random.seed(time.time())
    seed = random.randint(1,100)
    # setup_seed(20)
    # path = '..\data\img\grb_img\ivy-2.0\\buggy\ivy-2.0_src_java_org_apache_ivy_core_IvyPatternHelper.png'
    # # path = 'F:\Document\GitHub\DTLDP_master\data\img\grb_img\ivy-2.0\clean\ivy-2.0_src_java_org_apache_ivy_ant_AddPathTask.png'
    # with open(path, 'rb') as f:  # 以二进制格式打开一个文件用于只读
    #     with Image.open(f) as img:
    #         a = img.convert('RGB')

    path = '../data/txt_png_path/'
    # path = '../data/txt/'

    # Case1: 使用命令行
    # # 创建一个解析对象，对命令行参数进行解析 （这种方式不利于调试，可以对args进行赋值）
    # parser = argparse.ArgumentParser(description='Transfer Learning')
    # # 添加参数
    # parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    # parser.add_argument('--source', type=str, nargs='?', default='ant-1.6', help="source data")
    # parser.add_argument('--target', type=str, nargs='?', default='ant-1.7', help="target data")
    # parser.add_argument('--loss_name', type=str, nargs='?', default='DAN', help="loss name")
    # parser.add_argument('--tradeoff', type=float, nargs='?', default=1.0, help="tradeoff")
    # parser.add_argument('--using_bottleneck', type=int, nargs='?', default=1, help="whether to use bottleneck")
    # parser.add_argument('--task', type=str, nargs='?', default='WPDP', help="WPDP or CPDP")
    # # 解析
    # args = parser.parse_args()

    # Case2: 不使用命令行
    strings = ["ant-1.3", "camel-1.6", "ivy-2.0", "jedit-4.1", "log4j-1.2", "poi-2.0", "velocity-1.4", "xalan-2.4",
               "xerces-1.2"]
    new_arr = []
    test_arr = []

    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            new_arr.append(strings[i] + "->" + strings[j])
            new_arr.append(strings[j] + "->" + strings[i])


    parser = argparse.ArgumentParser(description='Transfer Learning')
    args = parser.parse_args()
    args.gpu_id = '0'
    args.source = 'xalan-2.4'
    args.target = 'xalan-2.4'
    args.loss_name = 'DAN'
    args.tradeoff = 1.0
    args.using_bottleneck = 0
    args.task = 'CPDP'  # 'WPDP' or 'CPDP'
    # cpdp 表示跨项目缺陷预测

    for i in range(len(new_arr)):
        args.source = new_arr[i].split("->")[0]
        args.target = new_arr[i].split("->")[1]
        mytarget_path = "../data/txt/" + args.target + ".txt"
        classnum = 1
        print(args.source+" "+args.target+" ", end='')
        print(classnum)

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

        # 定义一个字典类型变量
        config = {}
        # 添加键值对
        config["num_iterations"] = 10
        config["test_interval"] = 1  # ?
        # test_10crop 是一个布尔类型的参数，用于表示在测试集上是否进行 10-crop 测试。10-crop 测试是指在测试时将一张图片切成 10 个部分并对每个部分进行预测，然后将这 10 个预测结果进行平均或投票得到最终的预测结果。这种方法可以提高模型的准确性，特别是在处理图像数据时。
        config["prep"] = [{"name": "source", "type": "image", "test_10crop": False, "resize_size": 256, "crop_size": 224},
                          {"name": "target", "type": "image", "test_10crop": False, "resize_size": 256, "crop_size": 224}]
        config["loss"] = {"name": args.loss_name, "trade_off": args.tradeoff}
        #
        config["data"] = [{"name": "source", "type": "image", "list_path": {"train": path + args.source + ".txt"},
                           "batch_size": {"train": 64, "test": 64}},
                          {"name": "target", "type": "image", "list_path": {"train": path + args.target + ".txt"},
                           "batch_size": {"train": 64, "test": 64}}]
        config["network"] = {"name": "AlexNet", "use_bottleneck": args.using_bottleneck, "bottleneck_dim": 256}
        config["optimizer"] = {"type": "SGD",
                               "optim_params": {"lr": 0.05, "momentum": 0.9, "weight_decay": 0.0005, "nesterov": True},
                               "lr_type": "inv", "lr_param": {"init_lr": 0.0003, "gamma": 0.0003, "power": 0.75}}
        # 对代码的修改和理解  都吧注释写满  方便组员学习
        # num_iterations表示训练的迭代次数；
        # test_interval表示每多少个迭代进行一次测试；
        # prep表示数据预处理的配置，包括source和target两个来源的数据，需要进行的操作包括图片的缩放和裁剪；
        # loss表示损失函数的配置，包括使用的损失函数的名称和对各项损失的权重；
        # data表示训练和测试数据的配置，包括source和target两个来源的数据，需要读取的文件路径和每个batch的大小；
        # network表示神经网络的配置，包括使用的网络名称、是否使用bottleneck特征、bottleneck的维度等；
        # optimizer表示优化器的配置，包括使用的优化算法、学习率、动量、权重衰减等参数。
        test_result = transfer_classification(config,classnum,seed)
        print(new_arr[i],end=' ')
        print(" test", end=' ')
        print(test_result)

        # print(new_arr[i]+" train "+ train_result+" test"+test_result)
        test_arr.append(test_result)

    workbook = openpyxl.Workbook()
    # 选择默认的工作表
    worksheet = workbook.active

    for i in range(len(new_arr)):
        worksheet.cell(row=i + 1, column=1, value=new_arr[i])
        worksheet.cell(row=i + 1, column=2, value=test_arr[i])


    # 保存文件
    workbook.save('output_randomTree.xlsx')#运行失败 需要改一个别的文件名


# 让模型初始化的时候随机 就是说同一批train的数据生产的模型会略有不同
# 这样结果才会有 0.28 和0、27888 这种微小差别 才可以取平均值
