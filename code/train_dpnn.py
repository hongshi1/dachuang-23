import openpyxl

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from PerformanceMeasure import Origin_PerformanceMeasure as PerformanceMeasure
import random
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from PerformanceMeasure import Origin_PerformanceMeasure as PerformanceMeasure
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, dim_feedforward=dim_feedforward)
        self.pos_encoder = nn.Embedding(5000, d_model)
        self.d_model = d_model

    def forward(self, x):
        pos = torch.arange(0, x.size(1)).unsqueeze(0).to(x.device)
        x = x * np.sqrt(self.d_model) + self.pos_encoder(pos)
        return self.transformer.encoder(x)


class TransformerRegressor(nn.Module):
    def __init__(self, ch_in, d_model=256, nhead=4, num_encoder_layers=2, dim_feedforward=512):
        super(TransformerRegressor, self).__init__()

        self.embedding = nn.Linear(ch_in, d_model)
        self.transformer = TransformerEncoder(d_model, nhead, num_encoder_layers, dim_feedforward)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)  # Transformer expects seq_len, batch, features
        x = self.transformer(x)
        x = x.mean(dim=0)  # Take mean across sequence length
        return self.fc(x)


def train(source, target, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load Source Data
    cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic',
            'cbm', 'amc', 'max_cc', 'avg_cc','loc']
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



    # Load Target Data
    target_file_path = f'../data/promise_csv/{target}.csv'
    target_data = pd.read_csv(target_file_path, usecols=cols)  # Columns D to W are 3 to 22
    target_features = target_data.iloc[:, :].values  # All columns except the last one
    label_data = pd.read_csv(target_file_path, usecols=['bug'])
    loc_data = pd.read_csv(target_file_path, usecols=['loc'])
    loc_labels = loc_data.iloc[:].values.flatten()
    target_labels = label_data.iloc[:].values.flatten()  # The last column

    source_features = torch.Tensor(source_features).to(device)
    source_labels = torch.Tensor(source_labels).to(device)
    target_features = torch.Tensor(target_features).to(device)

    model =  TransformerRegressor(20).to(device)
    optimizer = optim.Adam(model.parameters(),lr=0.0001)
    criterion = nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(source_features)
        loss = criterion(outputs, source_labels)
        loss.backward()
        optimizer.step()

    # Predict using the model
    model.eval()
    with torch.no_grad():
        predictions = model(target_features)

    # Convert predictions to numpy array for the PerformanceMeasure class
    predictions = predictions.cpu().numpy()

    per = PerformanceMeasure(target_labels, predictions, loc_labels)
    pofb = per.POPT()

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
        workbook.save('../output/dp_data/output_DPNN_popt_newData'+str(round+1)+'.xlsx')  # 保存的文件名也修改为对应模型的名字