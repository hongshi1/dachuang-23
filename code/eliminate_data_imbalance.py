import os
import random
import numpy as np
from smotend import SMOTEND
import pandas as pd
from scipy.io.arff import loadarff
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

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

def eliminate_data_imbalance(origin_data_path, target_data_path, seed, normalization=False):
    origin_data = pd.read_csv(origin_data_path, header=None).iloc[1:, :].astype(float)
    target_data = pd.read_csv(target_data_path, header=None).iloc[1:, :].astype(float)

    #对数据做归一化
    if normalization:
        scaler = MinMaxScaler()
        origin_data.iloc[:, :-1] = scaler.fit_transform(origin_data.iloc[:, :-1])
        target_data.iloc[:, :-1] = scaler.fit_transform(target_data.iloc[:, :-1])

    # origin_data, target_data = preprocess(origin_data, target_data)
    # origin_data.to_csv(origin_data_path, index=False, header=cols)
    # target_data.to_csv(target_data_path, index=False, header=cols)
    # print('已完成：', origin_data_path, target_data_path)
    ori_data_shse_x, ori_data_shse_y = SHSE(np.array(origin_data), seed)
    mat_data = np.hstack([ori_data_shse_x, ori_data_shse_y.reshape(-1, 1)])
    return mat_data, target_data

if __name__=='__main__':
    strings = ["ant-1.3", "camel-1.6", "ivy-2.0", "jedit-4.1", "log4j-1.2", "poi-2.0", "velocity-1.4", "xalan-2.4",
               "xerces-1.2"]
#     cols='wmc	dit	noc	cbo	rfc	lcom	ca	ce	npm	lcom3	loc	dam	moa	mfa	cam	ic	cbm	amc	max_cc	avg_cc	ast_data_1	ast_data_2	ast_data_3	ast_data_4	ast_data_5	ast_data_6	ast_data_7	ast_data_8	ast_data_9	ast_data_10	ast_data_11	ast_data_12	ast_data_13	ast_data_14	ast_data_15	ast_data_16	ast_data_17	ast_data_18	ast_data_19	ast_data_20	ast_data_21	ast_data_22	ast_data_23	ast_data_24	ast_data_25	ast_data_26	ast_data_27	ast_data_28	ast_data_29	ast_data_30	ast_data_31	ast_data_32	ast_data_33	ast_data_34	ast_data_35	ast_data_36	ast_data_37	ast_data_38	ast_data_39	ast_data_40	ast_data_41	ast_data_42	ast_data_43	ast_data_44	ast_data_45	ast_data_46	ast_data_47	ast_data_48	ast_data_49	ast_data_50	ast_data_51	ast_data_52	ast_data_53	ast_data_54	ast_data_55	ast_data_56	ast_data_57	ast_data_58	ast_data_59	ast_data_60	ast_data_61	ast_data_62	ast_data_63	ast_data_64	ast_data_65	ast_data_66	ast_data_67	ast_data_68	ast_data_69	ast_data_70	ast_data_71	ast_data_72	ast_data_73	ast_data_74	ast_data_75	ast_data_76	ast_data_77	ast_data_78	ast_data_79	ast_data_80	ast_data_81	ast_data_82	ast_data_83	ast_data_84	ast_data_85	ast_data_86	ast_data_87	ast_data_88	ast_data_89	ast_data_90	ast_data_91	ast_data_92	ast_data_93	ast_data_94	ast_data_95	ast_data_96	ast_data_97	ast_data_98	ast_data_99	ast_data_100	imgVec_data_1	imgVec_data_2	imgVec_data_3	imgVec_data_4	imgVec_data_5	imgVec_data_6	imgVec_data_7	imgVec_data_8	imgVec_data_9	imgVec_data_10	imgVec_data_11	imgVec_data_12	imgVec_data_13	imgVec_data_14	imgVec_data_15	imgVec_data_16	imgVec_data_17	imgVec_data_18	imgVec_data_19	imgVec_data_20	imgVec_data_21	imgVec_data_22	imgVec_data_23	imgVec_data_24	imgVec_data_25	imgVec_data_26	imgVec_data_27	imgVec_data_28	imgVec_data_29	imgVec_data_30	imgVec_data_31	imgVec_data_32	imgVec_data_33	imgVec_data_34	imgVec_data_35	imgVec_data_36	imgVec_data_37	imgVec_data_38	imgVec_data_39	imgVec_data_40	imgVec_data_41	imgVec_data_42	imgVec_data_43	imgVec_data_44	imgVec_data_45	imgVec_data_46	imgVec_data_47	imgVec_data_48	imgVec_data_49	imgVec_data_50	imgVec_data_51	imgVec_data_52	imgVec_data_53	imgVec_data_54	imgVec_data_55	imgVec_data_56	imgVec_data_57	imgVec_data_58	imgVec_data_59	imgVec_data_60	imgVec_data_61	imgVec_data_62	imgVec_data_63	imgVec_data_64	imgVec_data_65	imgVec_data_66	imgVec_data_67	imgVec_data_68	imgVec_data_69	imgVec_data_70	imgVec_data_71	imgVec_data_72	imgVec_data_73	imgVec_data_74	imgVec_data_75	imgVec_data_76	imgVec_data_77	imgVec_data_78	imgVec_data_79	imgVec_data_80	imgVec_data_81	imgVec_data_82	imgVec_data_83	imgVec_data_84	imgVec_data_85	imgVec_data_86	imgVec_data_87	imgVec_data_88	imgVec_data_89	imgVec_data_90	imgVec_data_91	imgVec_data_92	imgVec_data_93	imgVec_data_94	imgVec_data_95	imgVec_data_96	imgVec_data_97	imgVec_data_98	imgVec_data_99	imgVec_data_100	imgVec_data_101	imgVec_data_102	imgVec_data_103	imgVec_data_104	imgVec_data_105	imgVec_data_106	imgVec_data_107	imgVec_data_108	imgVec_data_109	imgVec_data_110	imgVec_data_111	imgVec_data_112	imgVec_data_113	imgVec_data_114	imgVec_data_115	imgVec_data_116	imgVec_data_117	imgVec_data_118	imgVec_data_119	imgVec_data_120	imgVec_data_121	imgVec_data_122	imgVec_data_123	imgVec_data_124	imgVec_data_125	imgVec_data_126	imgVec_data_127	imgVec_data_128	bug
# '
    cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'loc', 'dam', 'moa', 'mfa', 'cam',
                'ic',
                'cbm', 'amc', 'max_cc', 'avg_cc']
    for i in range(100):
        cols.append('ast_data_'+str(i+1))
    for i in range(128):
        cols.append('imgVec_data_'+str(i+1))
    cols.append('bug')
    i = 7
    while i < len(strings):
        origin_path = '../data/promise_csv/'+strings[i]+'.csv'
        target_path = '../data/promise_csv/'+strings[i+1]+'.csv'
        i+=2
        eliminate_data_imbalance(origin_path, target_path, 0, cols)
    # source_data, target_data = eliminate_data_imbalance('../data/promise_csv/camel-1.6.csv', '../data/promise_csv/ant-1.3.csv', 1)
    # print(len(source_data), len(target_data))

