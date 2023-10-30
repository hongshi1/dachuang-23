from scipy.io import loadmat
import numpy as np
#我写的用来展示mat文件的工具类
# Load the .mat file
data = loadmat('D:\\PROJECT\\dachuang\\dachuang\\data\\embedding_mat\\ant-1.3\\buggy\\ant-1.3_src_java_org_apache_tools_ant_DefaultLogger.mat')


# Print the content
for key, value in data.items():
    if not key.startswith("__"):  # Ignore metadata
        print(key, ":", value)
