import os
import random
import numpy as np
from smotend import SMOTEND
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.impute import SimpleImputer

def preprocess(origin_data,target_data):
    # Replace missing values with mean
    imputer = SimpleImputer(strategy='mean')
    origin_data.iloc[:, :-1] = imputer.fit_transform(origin_data.iloc[:, :-1])
    target_data.iloc[:, :-1] = imputer.fit_transform(target_data.iloc[:, :-1])

    # 去除重复实例
    ori_dataOnly = origin_data.drop_duplicates().reset_index(drop=True)
    target_dataOnly = target_data.drop_duplicates().reset_index(drop=True)

    #去除不一致实例
    mat0_x = ori_dataOnly.iloc[:, :-1]
    idx = []
    for i in range(len(mat0_x) - 1):
        for j in range(i + 1, len(mat0_x)):
            if np.array_equal(mat0_x.iloc[i, :], mat0_x.iloc[j, :]) and ori_dataOnly.iloc[i, -1] != ori_dataOnly.iloc[j, -1]:
                idx.extend([i, j])
    idx = np.unique(idx)
    ori_dataOnly = ori_dataOnly.drop(idx).reset_index(drop=True)

    mat0_y = target_dataOnly.iloc[:, :-1]
    idx = []
    for i in range(len(mat0_y) - 1):
        for j in range(i + 1, len(mat0_y)):
            if np.array_equal(mat0_y.iloc[i, :], mat0_y.iloc[j, :]) and target_dataOnly.iloc[i, -1] != target_dataOnly.iloc[j, -1]:
                idx.extend([i, j])
    idx = np.unique(idx)
    target_dataOnly = target_dataOnly.drop(idx).reset_index(drop=True)

    # 删除loc为0的实例
    loc_idx = 9 if ori_dataOnly.shape[1] == 18 else 10
    ori_dataOnly = ori_dataOnly[ori_dataOnly.iloc[:, loc_idx] != 0.00000]
    target_dataOnly = target_dataOnly[target_dataOnly.iloc[:, loc_idx] != 0]

    # 打乱数据集
    # ori_dataOnly = ori_dataOnly.sample(frac=1).reset_index(drop=True)
    # target_dataOnly = target_dataOnly.sample(frac=1).reset_index(drop=True)
    return ori_dataOnly, target_dataOnly

def SHSE(train_data, fea_ratio=3/4, seed=0):
    # 默认值
    ins_ratio = 0
    ratio_def = np.sum(train_data[:, -1] > 0) / train_data.shape[0]
    if ratio_def < 0.5:
        if ratio_def + ratio_def * 0.5 <= 0.5:
            ins_ratio = (ratio_def + ratio_def * 0.5) / (1 - ratio_def)
        else:
            ins_ratio = 0.5 / (1 - ratio_def)
    else:
        ins_ratio = 1

    train_x_ori = train_data[:, :-1]
    train_y_ori = train_data[:, -1]

    #开始下采样
    #对正样本和负样本进行划分
    sel_fea_num = int(np.floor(train_x_ori.shape[1] * fea_ratio))       #确定需要的特征数
    idx_pos = (train_y_ori > 0)
    train_x_pos = train_x_ori[idx_pos, :]
    train_y_pos = train_y_ori[idx_pos]

    train_x_neg = train_x_ori[~idx_pos, :]
    train_y_neg = train_y_ori[~idx_pos]

    np.random.seed(seed)
    idx_sel_fea = np.random.permutation(train_x_ori.shape[1])[:sel_fea_num]  #随机选择一定数量的特征

    idx_sel_ins_pos_num = train_x_pos.shape[0]                              #计算正例和负例的选择数量
    idx_sel_ins_neg_num = int(np.floor(train_x_neg.shape[0] * ins_ratio))

    idx_pos = np.random.permutation(train_x_pos.shape[0])[:idx_sel_ins_pos_num]             #随机选择正例和负例的样本
    idx_neg = np.random.permutation(train_x_neg.shape[0])[:idx_sel_ins_neg_num]

    new_train_x = np.vstack([train_x_pos[idx_pos], train_x_neg[idx_neg]])             #使用随机选择的特征和样本构建新的训练集
    new_train_y = np.concatenate([train_y_pos[idx_pos], train_y_neg[idx_neg]])

    mat_data = np.hstack([new_train_x, new_train_y.reshape(-1, 1)])                 #使用随机选择的特征和样本构建新的训练集
    mat_data = np.unique(mat_data, axis=0)                                  #移除重复的实例。

    new_train_x = mat_data[:, :-1]                      #将处理后的数据分开为新的训练特征和标签。
    new_train_y = mat_data[:, -1]

    # SMOTEND 上采样
    syn_mino = SMOTEND(mat_data)

    if syn_mino.shape[0] > 0:
        new_train_x_bal = np.vstack([new_train_x, syn_mino[:, :-1]])
        new_train_y_bal = np.concatenate([new_train_y, syn_mino[:, -1]])
    else:
        new_train_x_bal = new_train_x
        new_train_y_bal = new_train_y

    return new_train_x_bal, new_train_y_bal

def eliminate_data_imbalance(origin_data_path, target_data_path, seed):
    origin_data = pd.read_csv(origin_data_path, header=None).iloc[1:, 3:].astype(float)
    target_data = pd.read_csv(target_data_path, header=None).iloc[1:, 3:].astype(float)

    origin_data, target_data = preprocess(origin_data, target_data)

    ori_data_shse_x, ori_data_shse_y = SHSE(np.array(origin_data), seed)
    mat_data = np.hstack([ori_data_shse_x, ori_data_shse_y.reshape(-1, 1)])
    return mat_data, target_data

if __name__=='__main__':
    source_data, target_data = eliminate_data_imbalance('../data/promise_csv/camel-1.6.csv', '../data/promise_csv/ant-1.3.csv', 1)
    print(len(source_data), len(target_data))

