'''
coding:utf-8
@ Description:
@ Time:2024/7/28 18:13
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import eigh

# 加载Facebook图数据，假设图的文件名为 'facebook_graph.edgelist'
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

# 获取图的节点数量和邻接矩阵
n = G.number_of_nodes()
A = nx.to_numpy_array(G)

# 计算度矩阵和度数对角矩阵
degrees = A.sum(axis=1)
D = np.diag(degrees)

# 计算拉普拉斯矩阵和归一化拉普拉斯矩阵
L = D - A
D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees))
L_norm = D_inv_sqrt @ L @ D_inv_sqrt

# 计算归一化拉普拉斯矩阵的前k小特征值和特征向量
k = 10  # 分区数量
eigvals, eigvecs = eigh(L_norm, subset_by_index=[0, k-1])

# 使用k个最小特征向量进行k-means聚类以确定社区划分
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=k)
kmeans.fit(eigvecs)
labels = kmeans.labels_

# 将社区划分结果转换为社区列表格式
communities = []
for i in range(k):
    community = [node for node, label in enumerate(labels) if label == i]
    communities.append(community)

# 创建社区映射
community_map = {node: i for i, community in enumerate(communities) for node in community}

# 创建社区连接矩阵
community_matrix = np.zeros((k, k), dtype=int)

# 计算社区内和社区间的边数
for edge in G.edges():
    u, v = edge
    if community_map[u] == community_map[v]:
        community_matrix[community_map[u], community_map[v]] += 1
    else:
        community_matrix[community_map[u], community_map[v]] += 1
        community_matrix[community_map[v], community_map[u]] += 1

# 计算社区内和社区间的最大可能边数
max_edges = np.zeros((k, k), dtype=int)
for i in range(k):
    size_i = len([node for node in community_map if community_map[node] == i])
    max_edges[i, i] = size_i * (size_i - 1) // 2
    for j in range(i + 1, k):
        size_j = len([node for node in community_map if community_map[node] == j])
        max_edges[i, j] = max_edges[j, i] = size_i * size_j

# 计算连接密度
density_matrix = np.divide(community_matrix, max_edges, out=np.zeros_like(community_matrix, dtype=float), where=max_edges!=0)

# 生成热力图
plt.figure(figsize=(10, 8))
sns.heatmap(density_matrix, annot=True, cmap="YlGnBu", fmt=".3f")
# plt.title('Community Intra and Inter Connection Heatmap',fontdict={'family': 'serif', 'fontname': 'Times New Roman'})
plt.title('Community Intra and Inter Connection Heatmap')
plt.xlabel('')
plt.ylabel('')
plt.show()
# n = G.number_of_nodes()
# A = nx.to_numpy_array(G)
#
# # 计算度矩阵和度数对角矩阵
# degrees = A.sum(axis=1)
# D = np.diag(degrees)
#
# # 计算拉普拉斯矩阵和归一化拉普拉斯矩阵
# L = D - A
# D_inv_sqrt = np.diag(1.0 / np.sqrt(degrees, where=degrees!=0))  # 防止除以零
# L_norm = D_inv_sqrt @ L @ D_inv_sqrt
#
# # 计算归一化拉普拉斯矩阵的前k小特征值和特征向量
# k = 10  # 分区数量
# eigvals, eigvecs = eigh(L_norm, subset_by_index=[0, k-1])
#
# # 使用k个最小特征向量进行k-means聚类以确定社区划分
# from sklearn.cluster import KMeans
# kmeans = KMeans(n_clusters=k)
# kmeans.fit(eigvecs)
# labels = kmeans.labels_
#
# # 将社区划分结果转换为社区列表格式
# communities = []
# for i in range(k):
#     community = [node for node, label in enumerate(labels) if label == i]
#     communities.append(community)
#
# # 创建社区映射
# community_map = {node: i for i, community in enumerate(communities) for node in community}
#
# # 创建社区连接矩阵
# community_matrix = np.zeros((k, k), dtype=int)
#
# # 计算社区内和社区间的边数
# for u, v in G.edges():
#     if community_map[u] == community_map[v]:
#         community_matrix[community_map[u], community_map[v]] += 1
#     else:
#         community_matrix[community_map[u], community_map[v]] += 1
#         community_matrix[community_map[v], community_map[u]] += 1
#
# # 计算社区内和社区间的最大可能边数
# max_edges = np.zeros((k, k), dtype=int)
# for i in range(k):
#     size_i = len(communities[i])
#     max_edges[i, i] = size_i * (size_i - 1) // 2  # 社区内最大可能边数
#     for j in range(i + 1, k):
#         size_j = len(communities[j])
#         max_edges[i, j] = max_edges[j, i] = size_i * size_j  # 社区间最大可能边数
#
# # 计算连接密度
# density_matrix = np.divide(community_matrix, max_edges, out=np.zeros_like(community_matrix, dtype=float), where=max_edges!=0)
#
# # 生成热力图
# plt.figure(figsize=(10, 8))
# sns.heatmap(density_matrix, annot=True, cmap="YlGnBu", fmt=".3f")
# plt.title('Community Intra and Inter Connection Heatmap')
# plt.xlabel('Community')
# plt.ylabel('Community')
# plt.show()