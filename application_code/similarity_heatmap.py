'''
coding:utf-8
@ Description:
@ Time:2024/8/1 0:56
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
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

# 计算归一化拉普拉斯矩阵
L = nx.normalized_laplacian_matrix(G).toarray()

# 计算特征值和特征向量
eigvals, eigvecs = np.linalg.eigh(L)

# 选择前k个最小的非零特征值对应的特征向量
k = 10  # 社区数
features = eigvecs[:, 1:k+1]  # 选择第2到第(k+1)个特征向量

# 使用KMeans进行聚类，按照特征向量的方向进行划分
kmeans = KMeans(n_clusters=k, random_state=0).fit(features)
labels = kmeans.labels_

# 分配节点到社区
partition = [[] for _ in range(k)]
for node, label in zip(G.nodes(), labels):
    partition[label].append(node)

# 计算社区间的杰卡德相似度
similarity_matrix = np.zeros((k, k))
for i, community_i in enumerate(partition):
    for j, community_j in enumerate(partition):
        intersection = len(set(community_i) & set(community_j))
        union = len(set(community_i) | set(community_j))
        similarity_matrix[i, j] = intersection / union

# 绘制社区相似性热力图
plt.figure(figsize=(8, 6))
plt.imshow(similarity_matrix, cmap='seismic', interpolation='nearest')
plt.colorbar()
plt.xlabel('Community Index', fontsize=14)
plt.ylabel('Community Index', fontsize=14)
plt.title('Community Similarity Heat Map', fontsize=16)
plt.show()