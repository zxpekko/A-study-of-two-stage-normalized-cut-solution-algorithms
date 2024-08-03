'''
coding:utf-8
@ Description:
@ Time:2024/7/4 21:03
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
from infomap import Infomap
import numpy as np
infomap = Infomap()
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

# 将图中的边添加到Infomap
for edge in G.edges():
    infomap.add_link(*edge)

# 运行Infomap算法
infomap.run()

# 获取社区划分结果
communities = infomap.get_modules()

# 获取社区数量
num_communities = len(set(communities.values()))

# 将社区划分结果转换为n×k的矩阵
n = len(G.nodes)
partition_matrix = np.zeros((n, num_communities), dtype=int)

for node, community in communities.items():
    partition_matrix[node, community - 1] = 1  # 社区编号从1开始，因此需要减1

# 保存分区矩阵到文件
np.save('partition_matrix.npy', partition_matrix)

# 输出社区划分结果
print(f"Number of communities: {num_communities}")
for i in range(num_communities):
    community_nodes = [node for node, comm in communities.items() if comm == i + 1]
    print(f"Community {i+1}: {community_nodes}")