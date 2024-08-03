'''
coding:utf-8
@ Description:
@ Time:2024/7/28 19:52
@ Author:蓝桉长在洱海边
'''
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# 假设 G 是已经加载的图
# G = nx.read_edgelist('facebook_graph.edgelist', nodetype=int)

# 获取拉普拉斯矩阵的特征向量
# L = nx.normalized_laplacian_matrix(G).toarray()
# eigvals, eigvecs = np.linalg.eigh(L)

# 初始化模块度列表
modularity_values = [0.0416,0.151,0.397,0.400,0.444,0.703,0.709,0.762,0.863,0.787,0.787,0.778,0.770,0.760,0.770,0.732,0.772,0.757,0.769]
partition_range = range(2, 21)  # 划分数量从2到20

# for k in partition_range:
#     # 使用前k个最小的非零特征值对应的特征向量（不包括第一个全0的特征向量）
#     features = eigvecs[:, 1:k+1]
#
#     # 使用KMeans进行聚类
#     kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
#     labels = kmeans.labels_
#
#     # 计算模块度
#     communities = [[] for _ in range(k)]
#     for node, label in zip(G.nodes(), labels):
#         communities[label].append(node)
#     modularity = nx.algorithms.community.quality.modularity(G, communities)
#     modularity_values.append(modularity)

# 绘制模块度随划分数量变化的折线图
plt.figure(figsize=(10, 6))
plt.plot(partition_range, modularity_values, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Partitions', fontsize=14)
plt.ylabel('Modularity', fontsize=14)
plt.title('Modularity vs. Number of Partitions', fontsize=16)
plt.grid(True)
plt.xticks(range(2, 21, 2))  # 设置x轴刻度仅显示2, 4, 6, ..., 20
plt.show()