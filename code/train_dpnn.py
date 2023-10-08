import openpyxl
from keras.models import Sequential
from keras.layers import Dense, Activation
import pandas as pd
from PerformanceMeasure import PerformanceMeasure
import random
import time
import numpy as np

class DPNN():
    def __init__(self, ch_in):
        self.ch_in = ch_in

    def get_model(self):
        # 创建一个Sequential模型
        model = Sequential()

        # 添加第一层，并设置权重初始化为均匀分布
        model.add(Dense(units=20, input_dim=self.ch_in, kernel_initializer='uniform'))
        model.add(Activation('tanh'))

        # 添加第二层，并设置权重初始化为均匀分布
        model.add(Dense(units=10, kernel_initializer='uniform'))
        model.add(Activation('relu'))

        # 添加第三层，并设置权重初始化为均匀分布
        model.add(Dense(units=6, kernel_initializer='uniform'))
        model.add(Activation('relu'))

        # 添加最后一层
        model.add(Dense(units=1, kernel_initializer='normal'))
        return model

def train(source, target, seed):
    # Load Source Data
    cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic',
            'cbm', 'amc', 'max_cc', 'avg_cc']
    source_file_path = f'../data/promise_csv/{source}.csv'
    source_data = pd.read_csv(source_file_path, usecols=cols)  # Columns D to W are 3 to 22
    source_features = source_data.iloc[:, :].values  # All columns except the last one
    label_data = pd.read_csv(source_file_path, usecols=['bug'])
    source_labels = label_data.iloc[:].values  # The last column

    np.random.seed(seed)
    # 打乱数据
    combined_data = list(zip(source_features, source_labels))
    np.random.shuffle(combined_data)

    # 分离打乱后的数据
    source_features, source_labels = zip(*combined_data)

    model = DPNN(19).get_model()
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(np.array(source_features), np.array(source_labels),epochs=50)

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

if __name__ == '__main__':
    for round in range(30):
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
            test_result = train(source, target, seed)
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
        workbook.save('C:/Users/lenovo/Desktop/dp_data/output_DPNN_pofb_'+str(round+1)+'.xlsx')  # 保存的文件名也修改为对应模型的名字