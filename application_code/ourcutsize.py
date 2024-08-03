'''
coding:utf-8
@ Description:
@ Time:2024/7/14 19:57
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans

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

# 将标签转换为社区
partition_matrix = np.zeros((len(G.nodes), k), dtype=int)

for node, label in enumerate(labels):
    partition_matrix[node, label] = 1

# 计算切割大小
cut_size = 0

# 遍历所有节点对，计算跨社区的边数
for i in range(len(G.nodes)):
    for j in range(i + 1, len(G.nodes)):
        if G.has_edge(i, j) and labels[i] != labels[j]:
            cut_size += 1

print(f"Cut Size: {cut_size}")

# 保存分区矩阵到文件
np.save('partition_matrix.npy', partition_matrix)

print(f"Partition matrix saved with shape: {partition_matrix.shape}")