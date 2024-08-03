'''
coding:utf-8
@ Description:
@ Time:2024/7/21 21:51
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community import modularity
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

# 使用Girvan-Newman算法进行社区检测
comp = girvan_newman(G)

# 获取第一个划分的结果
community_structure = next(comp)

# 转换社区结果为NetworkX可接受的格式
community_list = [list(community) for community in community_structure]

# 计算模块度
modularity_value = modularity(G, community_list)

print(f"Modularity: {modularity_value}")

# 保存分区矩阵到文件
n = G.number_of_nodes()
k = len(community_structure)
partition_matrix = np.zeros((n, k), dtype=int)
for community_id, community in enumerate(community_structure):
    for node in community:
        partition_matrix[node, community_id] = 1

np.save('partition_matrix_girvan_newman.npy', partition_matrix)
print(f"Partition matrix saved with shape: {partition_matrix.shape}")