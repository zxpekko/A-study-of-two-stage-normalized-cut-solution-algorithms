'''
coding:utf-8
@ Description:
@ Time:2024/6/30 14:13
@ Author:蓝桉长在洱海边
'''
import utils.solve_max_cut as solve_max_cut
import pickle
import networkx as nx
import numpy as np
import community as community_louvain
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
# print(type(G))
# print(G)
bitstring=solve_max_cut.solve_normalized_cut(G)
print(sum(bitstring[:,0]))
print(bitstring.shape)
# 创建一个分区字典，将节点映射到它们所属的分区
partition = {}
partition_matrix=bitstring
for i in range(partition_matrix.shape[0]):
    # 第一个分区的节点
    if partition_matrix[i, 0] == 1:
        partition[i] = 0
    # 第二个分区的节点
    elif partition_matrix[i, 1] == 1:
        partition[i] = 1

# 使用 NetworkX 提供的模块度函数计算模块度
communities = [set() for _ in range(2)]  # 假设有两个分区
for node, community in partition.items():
    communities[community].add(node)

# modularity = nx.algorithms.community.quality.modularity(G, communities)
# print(f"Modularity: {modularity}")
# 计算每个社区的内部密度
def calculate_internal_density(G, community):
    subgraph = G.subgraph(community)
    actual_edges = subgraph.number_of_edges()
    possible_edges = len(community) * (len(community) - 1) / 2  # 无向图
    if possible_edges == 0:
        return 0
    return actual_edges / possible_edges
internal_densities = []
# for community in communities:
#     internal_density = calculate_internal_density(G, community)
#     internal_densities.append(internal_density)

print(f"Internal Densities: {internal_densities}")


def calculate_boundary_density(G, partition):
    inter_edges = 0
    community_sizes = [0, 0]

    for u, v in G.edges():
        if partition[u] != partition[v]:
            inter_edges += 1

    for node in partition:
        community_sizes[partition[node]] += 1

    possible_inter_edges = community_sizes[0] * community_sizes[1]

    if possible_inter_edges == 0:
        return 0

    return inter_edges / possible_inter_edges
# boundary_density = calculate_boundary_density(G, partition)
# print(f"Boundary Density: {boundary_density}")
# 计算切割大小
def calculate_cut_size(G, partition):
    cut_size = 0
    for u, v in G.edges():
        if partition[u] != partition[v]:
            cut_size += 1
    return cut_size
cut_size = calculate_cut_size(G, partition)
print(f"Cut Size: {cut_size}")


def calculate_ratio_cut(G, partition):
    inter_edges = 0
    intra_edges = [0, 0]

    for u, v in G.edges():
        if partition[u] != partition[v]:
            inter_edges += 1
        else:
            intra_edges[partition[u]] += 1

    total_intra_edges = sum(intra_edges) / 2  # 每条内部边被计算了两次

    if total_intra_edges == 0:
        return float('inf')  # 避免除以零的情况

    return inter_edges / total_intra_edges


ratio_cut = calculate_ratio_cut(G, partition)
print(f"Ratio Cut: {ratio_cut}")


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