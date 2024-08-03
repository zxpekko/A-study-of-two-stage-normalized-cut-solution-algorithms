'''
coding:utf-8
@ Description:
@ Time:2024/7/1 14:07
@ Author:蓝桉长在洱海边
'''
import pickle

import networkx as nx
import community as community_louvain
import numpy as np
from itertools import combinations
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



# 使用Louvain算法进行社区检测
partition = community_louvain.best_partition(G)

# 获取社区的数量
num_communities = max(partition.values()) + 1

# 将分区结果转换为 n × k 的矩阵
n = len(partition)
partition_matrix = np.zeros((n, num_communities), dtype=int)

for node, community in partition.items():
    partition_matrix[node, community] = 1

# 保存分区矩阵
np.save('partition_matrix.npy', partition_matrix)

# 输出分区矩阵
print(partition_matrix)
# for i in range(partition_matrix.shape[1]):
#     print(sum(partition_matrix[:,i]))
partition = {}
for node in range(partition_matrix.shape[0]):
    community = np.where(partition_matrix[node] == 1)[0][0]
    partition[node] = community


# 计算模块度
def calculate_modularity(G, partition):
    m = G.size(weight='weight')  # 图中的总边数
    degrees = dict(G.degree(weight='weight'))  # 节点的度数
    modularity = 0.0

    for u in G.nodes():
        for v in G.nodes():
            if partition[u] == partition[v]:  # 检查节点是否在同一个社区
                A_uv = G[u][v]['weight'] if G.has_edge(u, v) else 0  # 边的权重
                modularity += A_uv - (degrees[u] * degrees[v]) / (2 * m)

    modularity /= (2 * m)
    return modularity


modularity = calculate_modularity(G, partition)
print(f"Modularity: {modularity}")
partition = {}
for node in range(partition_matrix.shape[0]):
    community = np.where(partition_matrix[node] == 1)[0][0]
    partition[node] = community

# 获取社区列表
communities = {}
for node, community in partition.items():
    if community not in communities:
        communities[community] = []
    communities[community].append(node)

# 计算社区内部边缘密度
def calculate_internal_edge_density(G, communities):
    internal_edges = 0
    possible_internal_edges = 0
    for community_nodes in communities.values():
        subgraph = G.subgraph(community_nodes)
        internal_edges += subgraph.size()
        possible_internal_edges += len(community_nodes) * (len(community_nodes) - 1) / 2
    internal_edge_density = internal_edges / possible_internal_edges if possible_internal_edges > 0 else 0
    return internal_edge_density

# 计算社区之间边缘密度
def calculate_external_edge_density(G, communities):
    external_edges = 0
    possible_external_edges = 0
    community_pairs = combinations(communities.keys(), 2)
    for c1, c2 in community_pairs:
        community_nodes1 = communities[c1]
        community_nodes2 = communities[c2]
        external_edges += sum(1 for node1 in community_nodes1 for node2 in community_nodes2 if G.has_edge(node1, node2))
        possible_external_edges += len(community_nodes1) * len(community_nodes2)
    external_edge_density = external_edges / possible_external_edges if possible_external_edges > 0 else 0
    return external_edge_density

internal_edge_density = calculate_internal_edge_density(G, communities)
external_edge_density = calculate_external_edge_density(G, communities)

print(f"Internal Edge Density: {internal_edge_density}")
print(f"External Edge Density: {external_edge_density}")
partition = {}
for node in range(partition_matrix.shape[0]):
    community = np.where(partition_matrix[node] == 1)[0][0]
    partition[node] = community

# 计算切割大小
def calculate_cut_size(G, partition):
    cut_size = 0
    for u, v in G.edges():
        if partition[u] != partition[v]:  # 如果边的两个节点属于不同社区
            cut_size += 1
    return cut_size

cut_size = calculate_cut_size(G, partition)
print(f"Cut Size: {cut_size}")
partition = {}
for node in range(partition_matrix.shape[0]):
    community = np.where(partition_matrix[node] == 1)[0][0]
    partition[node] = community


# 计算内部边数和外部边数
def calculate_internal_external_edges(G, partition):
    internal_edges = 0
    external_edges = 0

    for u, v in G.edges():
        if partition[u] == partition[v]:  # 如果边的两个节点属于同一社区
            internal_edges += 1
        else:  # 如果边的两个节点属于不同社区
            external_edges += 1

    return internal_edges, external_edges


# 计算内外比率
def calculate_internal_external_ratio(G, partition):
    internal_edges, external_edges = calculate_internal_external_edges(G, partition)
    ier = internal_edges / external_edges if external_edges > 0 else float('inf')
    return ier


internal_edges, external_edges = calculate_internal_external_edges(G, partition)
ier = calculate_internal_external_ratio(G, partition)

print(f"Internal Edges: {internal_edges}")
print(f"External Edges: {external_edges}")
print(f"Internal-External Ratio (IER): {ier}")

def calculate_davies_bouldin_index(G, partition):
    # 获取所有的社区
    communities = set(partition.values())

    # 计算每个社区的质心和散度
    centroids = {}
    dispersions = {}

    for community in communities:
        nodes = [node for node in partition if partition[node] == community]
        subgraph = G.subgraph(nodes)

        # 质心：节点编号的平均值
        centroids[community] = np.mean(nodes)

        # 散度：社区内部边的平均距离
        if subgraph.number_of_edges() > 0:
            dispersions[community] = np.mean([1 for _ in subgraph.edges()])
        else:
            dispersions[community] = 0

    # 计算社区之间的距离
    dbi = 0
    for i in communities:
        max_ratio = 0
        for j in communities:
            if i != j:
                distance = abs(centroids[i] - centroids[j])  # 使用质心的绝对差值作为距离
                ratio = (dispersions[i] + dispersions[j]) / distance
                if ratio > max_ratio:
                    max_ratio = ratio
        dbi += max_ratio

    dbi /= len(communities)
    return dbi


dbi = calculate_davies_bouldin_index(G, partition)
print(f"Davies-Bouldin Index: {dbi}")