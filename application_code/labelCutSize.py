'''
coding:utf-8
@ Description:
@ Time:2024/7/21 21:34
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np

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
communities = list(nx.community.label_propagation_communities(G))

# 将社区结果转换为标签
labels = np.zeros(len(G.nodes), dtype=int)
for community_id, community in enumerate(communities):
    for node in community:
        labels[node] = community_id

# 计算切割大小
cut_size = 0

# 遍历所有边，统计跨社区的边数
for u, v in G.edges():
    if labels[u] != labels[v]:
        cut_size += 1

print(f"Cut Size: {cut_size}")

# 保存分区矩阵到文件
k = len(communities)
partition_matrix = np.zeros((len(G.nodes), k), dtype=int)
for node, label in enumerate(labels):
    partition_matrix[node, label] = 1
np.save('partition_matrix.npy', partition_matrix)

print(f"Partition matrix saved with shape: {partition_matrix.shape}")