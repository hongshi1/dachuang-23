from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据集
import pandas as pd
import numpy as np

data_url = "http://lib.stat.cmu.edu/datasets/boston"                                #获取波士顿房价
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])                     #将波士顿房价数据集中的奇数行和偶数行取出做横向堆叠，作为测试数据集[::2, :]：从第1行开始，每隔1行取一次数据，取出来的是第1、3、5、...行的所有列。
y = raw_df.values[1::2, 2]                                                          # [1::2, :2]：从第2行开始，每隔1行取一次数据，取出来的是第2、4、6、...行的前2列

#在svr中测试数据集一般为二维数组，[n_smaples,n_features]  target为[n_samples]

# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)                   #test_size=0.2表示将数据集按8:2的比例分为训练集和测试集，random_state=42表示随机种子，以确保每次划分的结果一致。最终的输出为4个数组，分别为X_train, X_test, y_train, y_test，其中X_train和y_train是训练集的特征和目标变量，X_test和y_test是测试集的特征和目标变量。

# 设置超参数空间
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100], 'kernel': ['rbf']}

# 创建SVR模型
model = SVR()

# 使用网格搜索来确定最佳的超参数组合
grid = GridSearchCV(model, param_grid, cv=5)                    #sklearn中的一个超参数调优工具，用于帮助用户通过交叉验证来选择模型的最佳超参数。
grid.fit(X_train, y_train)                                      #做出拟合

# 使用最佳超参数来训练SVR模型
model = SVR(**grid.best_params_)
model.fit(X_train, y_train)

# 使用测试集来评估模型的性能
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)
