'''
coding:utf-8
@ Description:
@ Time:2024/7/14 21:51
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

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

# 计算归一化拉普拉斯矩阵
L = nx.normalized_laplacian_matrix(G).todense()

# 计算前10个最小的非零特征值对应的特征向量
k = 10  # 分成10个社区
eigenvalues, eigenvectors = eigsh(L, k=k+1, which='SM')

# 跳过第一个特征向量（对应特征值为0的特征向量）
X = eigenvectors[:, 1:k+1]

# 使用KMeans聚类
kmeans = KMeans(n_clusters=k, random_state=0).fit(X)
labels = kmeans.labels_

# 计算每个社区的中心
centroids = np.zeros((k, X.shape[1]))
for i in range(k):
    centroids[i, :] = np.mean(X[labels == i, :], axis=0)

# 计算每个社区的离散度
def cluster_dispersion(cluster_points, centroid):
    return np.mean(np.linalg.norm(cluster_points - centroid, axis=1))

dispersions = np.zeros(k)
for i in range(k):
    cluster_points = X[labels == i, :]
    dispersions[i] = cluster_dispersion(cluster_points, centroids[i, :])

# 计算达维尔指数
db_index = 0
for i in range(k):
    max_ratio = -1
    for j in range(k):
        if i != j:
            inter_cluster_distance = np.linalg.norm(centroids[i, :] - centroids[j, :])
            ratio = (dispersions[i] + dispersions[j]) / inter_cluster_distance
            if ratio > max_ratio:
                max_ratio = ratio
    db_index += max_ratio

db_index /= k

print(f"Davies-Bouldin Index: {db_index}")

# 保存分区矩阵到文件
partition_matrix = np.zeros((len(G.nodes), k), dtype=int)
for node, label in enumerate(labels):
    partition_matrix[node, label] = 1
np.save('partition_matrix.npy', partition_matrix)

print(f"Partition matrix saved with shape: {partition_matrix.shape}")