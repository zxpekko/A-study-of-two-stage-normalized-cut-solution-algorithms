'''
coding:utf-8
@ Description:
@ Time:2024/7/14 16:40
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
from networkx.algorithms.community.quality import modularity

def normalized_cut_spectral_clustering(G, num_clusters):
    # 计算拉普拉斯矩阵
    L = nx.normalized_laplacian_matrix(G).todense()

    # 计算拉普拉斯矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(L)

    # 选择前num_clusters个最小的非零特征值对应的特征向量
    X = eigenvectors[:, 1:num_clusters+1]

    # 使用KMeans聚类
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = kmeans.labels_

    return labels

# 加载Facebook图数据
file_path = '../graphs/facebook_combined.pkl'

# 打开PKL文件并加载内容
with open(file_path, 'rb') as file:
    data = pickle.load(file)
# print(type(data))
# print(data)
matrix=np.array(data)
# print(type(matrix))
# print(matrix[0])
G=nx.from_numpy_array(matrix[0])

# 重新编号节点，使得节点编号从0到n-1连续
G = nx.convert_node_labels_to_integers(G)

# 尝试不同的社区数量，计算每个数量下的模块度
max_clusters = 20
modularities = []
best_num_clusters = 2
best_modularity = -1
best_labels = None

for num_clusters in range(2, max_clusters + 1):
    labels = normalized_cut_spectral_clustering(G, num_clusters)

    # 将标签转换为社区
    communities = [[] for _ in range(num_clusters)]
    for node, label in enumerate(labels):
        communities[label].append(node)

    # 计算模块度
    current_modularity = modularity(G, communities)
    modularities.append(current_modularity)

    # 更新最佳模块度和最佳社区数量
    if current_modularity > best_modularity:
        best_modularity = current_modularity
        best_num_clusters = num_clusters
        best_labels = labels

print(f"Best number of clusters: {best_num_clusters}")
print(f"Best modularity: {best_modularity}")

# 输出最佳社区划分结果
partition_matrix = np.zeros((len(G.nodes), best_num_clusters), dtype=int)
for node, label in enumerate(best_labels):
    partition_matrix[node, label] = 1

# 保存分区矩阵到文件
np.save('partition_matrix.npy', partition_matrix)

print(f"Partition matrix saved with shape: {partition_matrix.shape}")
print(modularities)