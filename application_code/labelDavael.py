'''
coding:utf-8
@ Description:
@ Time:2024/7/14 22:32
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
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

# 使用Label Propagation Algorithm进行社区检测
# communities = nx.community.label_propagation_communities(G)

# 使用Label Propagation Algorithm进行社区检测
communities = list(nx.community.label_propagation_communities(G))

# 将社区结果转换为标签
labels = np.zeros(len(G.nodes), dtype=int)
for community_id, community in enumerate(communities):
    for node in community:
        labels[node] = community_id

# 获取社区数量
k = len(communities)

# 计算邻接矩阵
A = nx.adjacency_matrix(G).todense()

# 计算每个社区的中心（使用邻接矩阵中的节点连接情况来表示）
centroids = np.zeros((k, A.shape[1]))
for i in range(k):
    nodes_in_community = np.where(labels == i)[0]
    centroids[i, :] = np.mean(A[nodes_in_community], axis=0)

# 计算每个社区的离散度
def cluster_dispersion(cluster_points, centroid):
    return np.mean(np.linalg.norm(cluster_points - centroid, axis=1))

dispersions = np.zeros(k)
for i in range(k):
    nodes_in_community = np.where(labels == i)[0]
    cluster_points = A[nodes_in_community]
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