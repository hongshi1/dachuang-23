from itertools import combinations

from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
def project_cluster(n_clusters = 3):
    # Load Source Data
    strings = ["ant-1.3", "camel-1.6", "ivy-2.0", "jedit-4.1", "log4j-1.2", "poi-2.0", "velocity-1.4", "xalan-2.4",
               "xerces-1.2"]
    data = {}
    for source in strings:
        cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam', 'ic',
                'cbm', 'amc', 'max_cc', 'avg_cc']
        source_file_path = f'../data/promise_csv/{source}.csv'
        source_data = pd.read_csv(source_file_path, usecols=cols)  # Columns D to W are 3 to 2
        data[source] = source_data.mean()

    data_array = np.array(list(data.values()))
    kmeans = KMeans(n_clusters=n_clusters)
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
    distances = (distances-np.min(distances)) / (np.max(distances) - np.min(distances))

    # # 输出不同簇中心之间的距离
    # print("Distances between cluster centers:")
    # print(distances)
    cluster_labels = data['cluster']  # 从 data 字典中获取 'cluster' 数组
    project_clusters = {}

    # 使用项目名称作为键，将 'cluster' 数组中的值关联起来
    for i, project in enumerate(strings):
        project_clusters[project] = cluster_labels[i]

    return project_clusters, distances

if __name__=="__main__":
    project_cluster(4)