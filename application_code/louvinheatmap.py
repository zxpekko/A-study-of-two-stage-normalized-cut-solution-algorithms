'''
coding:utf-8
@ Description:
@ Time:2024/7/28 18:33
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import community as community_louvain  # Louvain算法的Python实现
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# 使用鲁汶算法进行社区划分
partition = community_louvain.best_partition(G)

# 获取社区数量和社区列表
communities = {}
for node, community in partition.items():
    if community not in communities:
        communities[community] = []
    communities[community].append(node)

num_communities = len(communities)

# 创建社区映射
community_map = {node: comm for comm, nodes in communities.items() for node in nodes}

# 创建社区连接矩阵
community_matrix = np.zeros((num_communities, num_communities), dtype=int)

# 计算社区内和社区间的边数
for u, v in G.edges():
    if community_map[u] == community_map[v]:
        community_matrix[community_map[u], community_map[v]] += 1
    else:
        community_matrix[community_map[u], community_map[v]] += 1
        community_matrix[community_map[v], community_map[u]] += 1

# 计算社区内和社区间的最大可能边数
max_edges = np.zeros((num_communities, num_communities), dtype=int)
for i, nodes_i in communities.items():
    size_i = len(nodes_i)
    max_edges[i, i] = size_i * (size_i - 1) // 2  # 社区内最大可能边数
    for j, nodes_j in communities.items():
        if i < j:
            size_j = len(nodes_j)
            max_edges[i, j] = max_edges[j, i] = size_i * size_j  # 社区间最大可能边数

# 计算连接密度
density_matrix = np.divide(community_matrix, max_edges, out=np.zeros_like(community_matrix, dtype=float), where=max_edges!=0)

# 生成热力图
plt.figure(figsize=(10, 8))
sns.heatmap(density_matrix, annot=True, cmap="YlGnBu", fmt=".2f")
plt.title('Community Intra and Inter Connection Heatmap (Louvain)')
plt.xlabel('Community')
plt.ylabel('Community')
plt.show()