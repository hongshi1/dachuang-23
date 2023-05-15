# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


# 创建数据集
np.random.seed(42)
X = np.random.rand(100, 1) * 6 - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建模型，设置参数（限制树的最大深度为2）（max_features，min_samples_leaf，min_samples_split，）
reg_tree = DecisionTreeRegressor(max_depth=2)

# 训练模型
reg_tree.fit(X_train, y_train)

# 使用模型进行预测
y_pred = reg_tree.predict(X_test)

# 计算均方误差（MSE）
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)



