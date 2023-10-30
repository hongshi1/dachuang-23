import os.path
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd

#常杰写的用于聚类的工具类文件

def replace_with_rank(matrix):
    flattened_matrix = matrix.flatten()
    unique_values, counts = np.unique(flattened_matrix, return_counts=True)
    rank = np.argsort(np.argsort(unique_values))
    rank_dict = dict(zip(unique_values, rank))

    replaced_matrix = np.vectorize(lambda x: {0: 0, 1: 10, 2: 100, 3: 1000}.get(rank_dict[x], x))(matrix)
    return replaced_matrix

def project_cluster(n_clusters = 3):
    # Load Source Data
    strings = ["ant-1.3", "ant-1.4", "ant-1.5", "ant-1.6", "ant-1.7", "camel-1.0", "camel-1.2", "camel-1.4", "camel-1.6", "ivy-1.0", "ivy-1.1", "ivy-1.2", "jedit-3.2", "jedit-4.0","jedit-4.1", "jedit-4.2", "jedit-4.3", "log4j-1.0", "log4j-1.1", "log4j-1.2", "poi-1.5", "poi-2.0", "poi-2.5", "poi-3.0", "velocity-1.4", "velocity-1.5","velocity-1.6", "xalan-2.4",
               "xalan-2.5", "xalan-2.6", "xalan-2.7", "xerces-1.1", "xerces-1.2", "xerces-1.3", "xerces-1.4"]
    data = {}
    filenames = ["ant", "camel", "ivy", "jedit", "log4j", "poi", "velocity", "xalan", "xerces"]
    data["feature"] = {}
    data["labels"] = {}
    for filename in filenames:
        for source in strings:
            cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam',
                    'ic',
                    'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
            source_file_path = f'../data/bug-data/{filename}/{source}.csv'
            if os.path.exists(source_file_path):
                source_data = pd.read_csv(source_file_path, usecols=cols)  # Columns D to W are 3 to 2
                data["feature"][source] = source_data.iloc[:, :-1].mean()
                data["labels"][source] = source_data.iloc[:, -1].mean()

    scaler = StandardScaler()
    data_array = scaler.fit_transform(np.array(list(data["feature"].values())))
    kmeans = KMeans(n_clusters=n_clusters,init='k-means++')
    data['cluster'] = kmeans.fit_predict(data_array)

    # 计算每个簇的中心
    cluster_centers = kmeans.cluster_centers_

    # 计算不同簇中心之间的距离
    n_clusters = len(cluster_centers)
    distances = np.zeros((n_clusters, n_clusters))
    for i, j in combinations(range(n_clusters), 2):         #将不同簇之间两两组合
        distance = np.linalg.norm(cluster_centers[i] - cluster_centers[j])
        distances[i, j] = distance
        distances[j, i] = distance


    # # 归一化
    distances = replace_with_rank(distances)
    # distances = (distances-np.min(distances)) / (np.max(distances) - np.min(distances))
    # distances = distances + np.eye(n_clusters)*0.1
    # # 输出不同簇中心之间的距离
    # print("Distances between cluster centers:")
    # print(distances)
    cluster_labels = data['cluster']  # 从 data 字典中获取 'cluster' 数组
    project_clusters = {}
    # 使用项目名称作为键，将 'cluster' 数组中的值关联起来
    for i, project in enumerate(strings):
        project_clusters[project] = cluster_labels[i]

    # 使用项目名称作为键，将 'cluster' 数组中的值关联起来
    tsne_result = TSNE(n_components=2, learning_rate="auto").fit_transform(data_array)
    # 创建颜色映射
    colors = ['r', 'b', 'g', 'y']  # 可以根据需要添加更多颜色

    # 绘制散点图，为每个数据点分配相应的颜色
    for i in range(len(cluster_labels)):
        plt.scatter(tsne_result[i, 0], tsne_result[i, 1], c=colors[cluster_labels[i]])

    # 显示图例
    for i in range(n_clusters):
        plt.scatter([], [], c=colors[i], label=f'Cluster {i}')

    plt.legend()
    plt.title('t-SNE Visualization with K-Means Clusters')
    plt.show()
    return project_clusters, distances

if __name__=="__main__":
    project_cluster()