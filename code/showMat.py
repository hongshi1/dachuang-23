# from scipy.io import loadmat
# import numpy as np
#
# # Load the .mat file
# data = loadmat('D:\\PROJECT\\dachuang\\dachuang\\data\\embedding_mat\\ant-1.3\\buggy\\ant-1.3_src_java_org_apache_tools_ant_IntrospectionHelper.mat')
#
# # Print the content and shape
# for key, value in data.items():
#     if not key.startswith("__"):  # Ignore metadata
#         print(key, ":", value)
#         print(key, " shape:", np.shape(value))

import os
import numpy as np

def find_npy_files(directory):
    """
    递归查找指定目录及其子目录下的所有 .npy 文件。
    """
    npy_files = []
    # os.walk 递归遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                npy_files.append(os.path.join(root, file))
    return npy_files

def display_npy_files_data(directory):
    """
    显示指定目录下所有 .npy 文件的路径和数据。
    """
    npy_files = find_npy_files(directory)
    for file_path in npy_files:
        data = np.load(file_path)
        print(f"Path: {file_path}")
        print("Data: ", data)
        print("----\n")

# 替换这里的 'your_directory_path' 为你想要搜索的目录路径
your_directory_path = 'D:\\PROJECT\\dachuang\\dachuang\\data\\embedding'

# 调用函数
display_npy_files_data(your_directory_path)

