# encoding: utf-8

import openpyxl
from PerformanceMeasure import PerformanceMeasure
import torch.optim as optim

import random

import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # 新增引用RandomForestRegressor


def train(source, target):
    # Load Source Data
    cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic',
            'cbm', 'amc', 'max_cc', 'avg_cc']
    source_file_path = f'../data/promise_csv/{source}.csv'
    source_data = pd.read_csv(source_file_path, usecols=cols)  # Columns D to W are 3 to 22
    source_features = source_data.iloc[:, :].values  # All columns except the last one
    label_data = pd.read_csv(source_file_path, usecols=['bug'])
    source_labels = label_data.iloc[:].values  # The last column

    # Train a RandomForest Model using Source Data
    model = RandomForestRegressor()  # 这里修改为RandomForestRegressor
    model.fit(source_features, source_labels.ravel())  # 因为RandomForestRegressor期望y是一维的，所以这里使用ravel()

    # Load Target Data
    target_file_path = f'../data/promise_csv/{target}.csv'
    target_data = pd.read_csv(target_file_path, usecols=cols)  # Columns D to W are 3 to 22
    target_features = target_data.iloc[:, :].values  # All columns except the last one
    label_data = pd.read_csv(target_file_path, usecols=['bug'])
    loc_data = pd.read_csv(target_file_path, usecols=['loc'])
    loc_labels = loc_data.iloc[:].values.flatten()
    target_labels = label_data.iloc[:].values.flatten()  # The last column

    # Predict using the model and calculate MSE
    predictions = model.predict(target_features)
    per = PerformanceMeasure(target_labels, predictions, loc_labels)
    pofb = per.getPofb()

    # Return the MSE
    return pofb


if __name__ == "__main__":
    random.seed(time.time())
    seed = random.randint(1, 100)

    path = '../data/txt_png_path/'

    # Case2: 不使用命令行
    strings = ["ant-1.3", "camel-1.6", "ivy-2.0", "jedit-4.1", "log4j-1.2", "poi-2.0", "velocity-1.4", "xalan-2.4",
               "xerces-1.2"]
    new_arr = []
    test_arr = []

    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            new_arr.append(strings[i] + "->" + strings[j])
            new_arr.append(strings[j] + "->" + strings[i])

    for i in range(len(new_arr)):
        source = new_arr[i].split("->")[0]
        target = new_arr[i].split("->")[1]
        test_result = train(source, target)
        print(new_arr[i], end=' ')
        print(" test", end=' ')
        print(test_result)
        test_result = float(test_result)
        test_arr.append(test_result)

    workbook = openpyxl.Workbook()
    # 选择默认的工作表
    worksheet = workbook.active

    for i in range(len(new_arr)):
        worksheet.cell(row=i + 1, column=1, value=new_arr[i])
        worksheet.cell(row=i + 1, column=2, value=test_arr[i])

    # 保存文件
    workbook.save('../output/output_randomforest_pofb.xlsx')  # 保存的文件名也修改为对应RandomForest模型的名字
