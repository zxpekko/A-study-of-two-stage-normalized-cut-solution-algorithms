'''
coding:utf-8
@ Description:
@ Time:2024/7/21 22:05
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import numpy as np
import random
from networkx.algorithms.community import modularity


def karger_min_cut(G):
    """
    使用Karger's Min-Cut算法找到图G的一个最小割。
    """
    G_copy = G.copy()
    while len(G_copy.nodes()) > 2:
        u, v = random.choice(list(G_copy.edges()))
        G_copy = nx.contracted_nodes(G_copy, u, v, self_loops=False)

    return len(G_copy.edges()), G_copy


def community_detection_karger(G):
    """
    使用Karger's Min-Cut算法对图G进行社区检测。
    """
    num_nodes = len(G.nodes())
    min_cut_value, G_cut = karger_min_cut(G)

    # 通过最小割结果划分社区
    communities = [set(), set()]
    for node in G_cut.nodes():
        if node in G_cut.nodes():
            if len(G_cut.nodes()) == 2:
                communities[0].add(node)
            else:
                communities[1].add(node)

    return communities


def compute_modularity(G, communities):
    """
    计算给定社区划分的模块度。
    """
    return modularity(G, communities)


# 示例：创建一个Facebook图的示例图（需要用实际数据替换）
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

# 执行社区检测
communities = community_detection_karger(G)

# 计算模块度
mod = compute_modularity(G, communities)
print(f"Communities: {communities}")
print(f"Modularity: {mod}")