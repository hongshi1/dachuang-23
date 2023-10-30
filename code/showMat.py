from scipy.io import loadmat
import numpy as np

# Load the .mat file
data = loadmat('D:\\PROJECT\\dachuang\\dachuang\\data\\embedding_mat\\ant-1.3\\buggy\\ant-1.3_src_java_org_apache_tools_ant_IntrospectionHelper.mat')

# Print the content and shape
for key, value in data.items():
    if not key.startswith("__"):  # Ignore metadata
        print(key, ":", value)
        print(key, " shape:", np.shape(value))
