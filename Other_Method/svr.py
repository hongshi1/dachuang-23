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
# 标准化数据
scaler = StandardScaler()
X = scaler.fit_transform(X)         #标准化数据，使X的均值为0，标准差为1，可以避免某些特征对模型影响过大

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)     #将波士顿房价数据集中的奇数行和偶数行取出做横向堆叠，作为测试数据集[::2, :]：从第1行开始，每隔1行取一次数据，取出来的是第1、3、5、...行的所有列。


# 设置超参数空间
param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10, 100], 'kernel': ['rbf']}    #gamma用于决定模型在细节上的拟合程度，gamma的值越大，模型越倾向于在训练集上拟合每个样本的细节，这可能导致过拟合。而gamma的值越小，模型越倾向于拟合整个训练集的整体趋势，这可能导致欠拟合 C惩罚参数，越大对错误月放松，越小对错误的容忍度越小

# 创建SVR模型
model = SVR()

# 使用网格搜索来确定最佳的超参数组合
grid = GridSearchCV(model, param_grid, cv=5)   # sklearn中的一个超参数调优工具，用于帮助用户通过交叉验证来选择模型的最佳超参数。cv=5表示将数据分成5份，用四分来训练，一份来测试
grid.fit(X_train, y_train)

# 使用最佳超参数来训练SVR模型
model = SVR(**grid.best_params_)
model.fit(X_train, y_train)

# 使用测试集来评估模型的性能
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("MSE:", mse)

