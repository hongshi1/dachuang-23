import os.path
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AffinityPropagation
from numpy import unique



def replace_with_rank(matrix):
    flattened_matrix = matrix.flatten()
    unique_values, counts = np.unique(flattened_matrix, return_counts=True)
    rank = np.argsort(np.argsort(unique_values))
    rank_dict = dict(zip(unique_values, rank))

    replaced_matrix = np.vectorize(lambda x: {0: 0.1, 1: 0.2, 2: 0.3, 3: 0.4, 4: 0.5}.get(rank_dict[x], x))(matrix)
    return replaced_matrix

def project_cluster():
    # Load Source Data
    strings = ["ant-1.3", "camel-1.6", "ivy-2.0", "jedit-4.1", "log4j-1.2", "poi-2.0", "velocity-1.4", "xalan-2.4",
               "xerces-1.2"]
    data = {}
    data["feature"] = {}
    data["labels"] = {}
    for source in strings:
        cols = ['wmc', 'dit', 'noc', 'cbo', 'rfc', 'lcom', 'ca', 'ce', 'npm', 'lcom3', 'dam', 'moa', 'mfa', 'cam',
                'ic',
                'cbm', 'amc', 'max_cc', 'avg_cc', 'loc']
        source_file_path = f'../data/promise_csv/{source}.csv'
        if os.path.exists(source_file_path):
            source_data = pd.read_csv(source_file_path, usecols=cols)  # Columns D to W are 3 to 2
            data["feature"][source] = source_data.mean()

    np.random.seed(10)
    scaler = StandardScaler()
    data_array = scaler.fit_transform(np.array(list(data["feature"].values())))
    AP = AffinityPropagation(damping=0.5)
    AP.fit(data_array)
    train = AP.predict(data_array)
    clusters = unique(train)

    # 计算每个簇的估计中心
    cluster_center_indices = AP.cluster_centers_indices_
    cluster_centers = data_array[cluster_center_indices]  # 获取簇中心的数据点
    # 计算距离矩阵
    num_clusters = len(cluster_center_indices)
    distances = np.zeros((num_clusters, num_clusters))
    # 使用欧氏距离计算簇中心之间的距离
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):
            dist = np.linalg.norm(cluster_centers[i] - cluster_centers[j])  # 使用numpy的欧氏距离计算
            distances[i][j] = dist
            distances[j][i] = dist

    #做归一化
    min_val = np.min(distances)
    max_val = np.max(distances)
    distances = (distances - min_val)/(max_val - min_val)

    data_set = np.append(data_array, cluster_centers, axis=0)
    # 使用项目名称作为键，将 'cluster' 数组中的值关联起来
    project_clusters = {}
    for i, project in enumerate(strings):
        project_clusters[project] = train[i]

    # # 使用项目名称作为键，将 'cluster' 数组中的值关联起来
    # tsne_result = TSNE(n_components=2, learning_rate="auto", perplexity=5).fit_transform(data_set)
    # # 创建颜色映射
    # colors = ['r', 'b', 'g', 'y', 'c', 'm', 'k', 'orange', 'purple', 'pink',
    #           'dodgerblue', 'lime', 'indigo', 'goldenrod', 'brown', 'tomato',
    #           'peru', 'slategray', 'seagreen', 'deepskyblue', 'darkviolet']  # 可以根据需要添加更多颜色
    #
    # # 绘制散点图，为每个数据点分配相应的颜色
    # for i in range(len(train)):
    #     if i < len(train) - clusters.size:
    #         plt.scatter(tsne_result[i, 0], tsne_result[i, 1], c=colors[train[i]])
    #     else:
    #         plt.scatter(tsne_result[i, 0], tsne_result[i, 1], c=colors[len(colors)-1])
    #
    # # 显示图例
    # for i in range(clusters.size):
    #     plt.scatter([], [], c=colors[i], label=f'Cluster {i}')
    # plt.scatter([], [], c=colors[len(colors)-1], label=f'Cluster Century')
    #
    # plt.legend()
    # plt.title('t-SNE Visualization with K-Means Clusters')
    # plt.show()

    new_distances = replace_with_rank(distances)
    return project_clusters, new_distances


if __name__=='__main__':
    project_cluster()