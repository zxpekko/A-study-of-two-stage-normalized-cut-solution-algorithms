'''
coding:utf-8
@ Description:
@ Time:2024/8/4 1:19
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

# 统计每个社区的节点数量
community_sizes = np.bincount(labels)

# 绘制社区节点大小的柱状图
plt.figure(figsize=(10, 6))
plt.bar(range(k), community_sizes, color='skyblue')
plt.xlabel('Community Index', fontsize=14)
plt.ylabel('number of nodes', fontsize=14)
plt.title('number of nodes in each community', fontsize=16)
plt.xticks(range(k))
plt.show()