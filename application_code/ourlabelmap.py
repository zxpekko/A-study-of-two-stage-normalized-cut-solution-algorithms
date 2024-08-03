'''
coding:utf-8
@ Description:
@ Time:2024/7/28 19:39
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

# 假设 G 是已经加载的图
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

# 获取拉普拉斯矩阵的特征向量
L = nx.normalized_laplacian_matrix(G).toarray()
eigvals, eigvecs = np.linalg.eigh(L)

# 使用前k个最小的非零特征值对应的特征向量（不包括第一个全0的特征向量）
k = 10  # 社区数量
features = eigvecs[:, 1:k+1]  # 取出第2到第(k+1)个特征向量

# 使用KMeans进行聚类
kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
labels = kmeans.labels_

# 将节点分配到各个社区
node_to_community = {node: label for node, label in zip(G.nodes(), labels)}

# 找到度数最大的节点
max_degree_node = max(G.degree, key=lambda x: x[1])[0]

# 获取该节点及其邻居节点
neighbors = list(G.neighbors(max_degree_node))
subgraph_nodes = [max_degree_node] + neighbors
subgraph = G.subgraph(subgraph_nodes)

# 分配颜色
colors = list(mcolors.TABLEAU_COLORS)
color_map = [colors[node_to_community[node] % len(colors)] for node in subgraph.nodes()]

# 调整画布大小和布局
plt.figure(figsize=(10, 10))
pos = nx.spring_layout(subgraph, k=0.3)  # 增加节点间的距离
nx.draw_networkx_nodes(subgraph, pos, node_color=color_map, node_size=100, alpha=0.8)
nx.draw_networkx_edges(subgraph, pos, alpha=0.5, width=1)
nx.draw_networkx_labels(subgraph, pos, font_size=12)

plt.title(f"Node {max_degree_node} and Its Neighbors (Community Visualization)", fontdict={'fontsize': 16})
plt.show()

# 打印出邻居节点的社区归属情况
neighbor_communities = [node_to_community[neighbor] for neighbor in neighbors]
print(f"Max degree node: {max_degree_node}")
print(f"Neighbor community distribution: {neighbor_communities}")