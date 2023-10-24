import os.path
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from sklearn.manifold import TSNE, SpectralEmbedding
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean

def replace_with_rank(matrix):
    flattened_matrix = matrix.flatten()
    unique_values, counts = np.unique(flattened_matrix, return_counts=True)
    rank = np.argsort(np.argsort(unique_values))
    rank_dict = dict(zip(unique_values, rank))

    replaced_matrix = np.vectorize(lambda x: {0: 0, 1: 10, 2: 100, 3: 1000}.get(rank_dict[x], x))(matrix)
    return replaced_matrix

def project_cluster(n_clusters = 3):
    # Load Source Data
    strings = ["ant-1.3", "camel-1.6", "ivy-2.0", "jedit-4.1", "log4j-1.2", "poi-2.0", "velocity-1.4", "xalan-2.4",
               "xerces-1.2"]
    data = {}
    data["feature"] = {}
    data["labels"] = {}
    for source in strings:
        cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam',
                'ic',
                'cbm', 'amc', 'max_cc', 'avg_cc', 'bug']
        source_file_path = f'../data/promise_csv/{source}.csv'
        if os.path.exists(source_file_path):
            source_data = pd.read_csv(source_file_path, usecols=cols)  # Columns D to W are 3 to 2
            data["feature"][source] = source_data.iloc[:, :-1].mean()
            data["labels"][source] = source_data.iloc[:, -1].mean()

    scaler = StandardScaler()
    data_array = scaler.fit_transform(np.array(list(data["feature"].values())))

    #基于欧氏距离计算相似度矩阵
    similarity_matrix = np.zeros((len(strings), len(strings)))
    # 计算相似度矩阵
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            similarity_matrix[i, j] = euclidean(data_array[i], data_array[j])
            similarity_matrix[j, i] = similarity_matrix[i, j]  # 对称矩阵

    #构建加权图
    threshold = 0.5  ## 定义相似度阈值
    adj_matrix = np.zeros((len(strings), len(strings)))
    for i in range(len(strings)):
        for j in range(i + 1, len(strings)):
            if similarity_matrix[i, j] > threshold:
                adj_matrix[i, j] = similarity_matrix[i, j]
                adj_matrix[j, i] = similarity_matrix[i, j]  # 邻接矩阵是对称的

    # 4. 谱嵌入
    embedding = SpectralEmbedding(n_components=2)           #将数据降维到2维
    low_dimensional_data = embedding.fit_transform(adj_matrix)

    spectral_clustering = SpectralClustering(n_clusters=n_clusters)
    data['cluster'] = spectral_clustering.fit_predict(low_dimensional_data)

    # 计算每个簇的估计中心
    cluster_centers = []
    for cluster_id in range(n_clusters):
        cluster_data = low_dimensional_data[data["cluster"] == cluster_id]  # 提取属于当前簇的数据点
        cluster_center = np.mean(cluster_data, axis=0)  # 计算均值作为估计的中心
        cluster_centers.append(cluster_center)

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

    # # 使用项目名称作为键，将 'cluster' 数组中的值关联起来
    # tsne_result = TSNE(n_components=2, perplexity=5).fit_transform(low_dimensional_data)
    # # 创建颜色映射
    # colors = ['r', 'b', 'g', 'y']  # 可以根据需要添加更多颜色
    #
    # # 绘制散点图，为每个数据点分配相应的颜色
    # for i in range(len(cluster_labels)):
    #     plt.scatter(tsne_result[i, 0], tsne_result[i, 1], c=colors[cluster_labels[i]])
    #
    # # 显示图例
    # for i in range(n_clusters):
    #     plt.scatter([], [], c=colors[i], label=f'Cluster {i}')
    #
    # plt.legend()
    # plt.title('t-SNE Visualization with K-Means Clusters')
    # plt.show()

    return project_clusters, distances

if __name__=="__main__":
    project_cluster()