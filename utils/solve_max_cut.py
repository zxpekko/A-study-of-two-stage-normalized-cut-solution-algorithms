# import torch
import random
import os
import numpy as np
import networkx as nx

from collections import defaultdict

from scipy.linalg import eigh

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed_value = 1
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
# torch.manual_seed(seed_value)  # seed torch RNG

# TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TORCH_DTYPE = torch.float32
# print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')

def qubo_dict_to_ndarray(nx_G,Q):
    n_nodes=len(nx_G.nodes)
    Q_mat=np.zeros((n_nodes,n_nodes))
    for (x,y),value in Q.items():
        Q_mat[x][y]=value
    return  Q_mat

def gen_adj_dict(nx_G):
    Q_dict=defaultdict(int)
    for (u,v) in nx_G.edges:
        is_has_weight=len(list(nx_G.get_edge_data(u,v).values()))
        break
    if is_has_weight==0:
        for (u,v) in nx_G.edges:
            Q_dict[(u,v)]=1
            Q_dict[(v,u)]=1
    else:
        for (u,v) in nx_G.edges:
            val=list(nx_G.get_edge_data(u,v).values())[0]
            Q_dict[(u,v)]=val
            Q_dict[(v,u)]=val
    return Q_dict

def q_dict_to_array(nx_graph,Q):
    n_nodes=len(nx_graph.nodes)
    Q_mat=np.zeros((n_nodes,n_nodes))
    for (x,y),val in Q.items():
        Q_mat[x][y]=val
    return Q_mat

def gen_degree_array(nx_graph,adj_array):
    n_nodes=len(nx_graph.nodes)
    degree_array=np.zeros((n_nodes,n_nodes))
    for i in range(n_nodes):
        degree=sum(adj_array[i,:])
        degree_array[i][i]=degree
    return degree_array

def gen_q_dict_max_cut(nx_G, localdegree=2):
    Q_dic=defaultdict(int)
    for (u,v) in nx_G.edges:
        hasweight=len(list(nx_G.get_edge_data(u,v).values()))
        break
    if(hasweight==0):
        for (u, v) in nx_G.edges:
            Q_dic[(u, v)] = localdegree
        for u in nx_G.nodes:
            Q_dic[(u, u)] = -nx_G.degree(u)
    else:
        for (u, v) in nx_G.edges:
            Q_dic[(u, v)] = localdegree*list(nx_G.get_edge_data(u,v).values())[0]
        for u in nx_G.nodes:
            diag_weight=0
            for (u1,v1) in nx_G.edges:
                if u==u1 or u==v1:
                    diag_weight+=list(nx_G.get_edge_data(u1,v1).values())[0]
            Q_dic[(u, u)] = -diag_weight
    return Q_dic

def postprocess_gnn_max_cut(best_bitstring,nx_graph):
    q_mat = gen_q_dict_max_cut(nx_graph)
    q_mat = qubo_dict_to_ndarray(nx_graph,q_mat)
    if type(best_bitstring)!=list:
        best_bitstring=best_bitstring.cpu().numpy()
        bitstring_list = list(best_bitstring)
    else:bitstring_list=best_bitstring
    size_max_cut=sum(bitstring_list)
    bitstring_array=np.asarray(bitstring_list).reshape(-1,1)
    cut_value=-bitstring_array.T@q_mat@bitstring_array
    cut_value=cut_value[0][0]
    cut_solution=set([node for node, entry in enumerate(bitstring_list) if entry == 1])
    return size_max_cut,cut_value,cut_solution

def sollist_to_bitstring(solver_max_cut_list,n_nodes):
    bitstring=[0]*n_nodes
    for i in range(len(solver_max_cut_list)):
        bitstring[solver_max_cut_list[i]]=1
    return bitstring

def solve_normalized_cut(nx_graph):
    Q_dict=gen_adj_dict(nx_graph)
    adjacency_matrix=q_dict_to_array(nx_graph,Q_dict)
    for i in range(adjacency_matrix.shape[0]):
        adjacency_matrix[i][i]=1
    d_i=np.sum(adjacency_matrix,axis=1)
    D=np.diag(d_i)
    _,eigenvectors=eigh(D-adjacency_matrix,D,subset_by_index=[1,2])
    eigenvec=np.copy(eigenvectors)
    second_smallest_vec=eigenvectors[:,0]
    avg=np.sum(second_smallest_vec)/len(second_smallest_vec)
    bipartition=second_smallest_vec>avg

    bitstring=np.zeros((bipartition.shape[0],2))
    for i in range(bitstring.shape[0]):
        if bipartition[i]==0:
            bitstring[i,0]=0
            bitstring[i,1]=1
        else:
            bitstring[i,0]=1
            bitstring[i,1]=0
    return bitstring

def postprocess_normalzed_cut(best_bitstring,nx_graph):
    adj_array=q_dict_to_array(nx_graph,gen_adj_dict(nx_graph))
    degree_array=gen_degree_array(nx_graph,adj_array)
    L=degree_array-adj_array
    ncut=0
    for i in range(best_bitstring.shape[1]):
        sub_bitstring=best_bitstring[:,i]
        sub_bitstring_=sub_bitstring.reshape(-1,1)
        multi = (sub_bitstring_.T @ degree_array @ sub_bitstring_) ** (-1 / 2)
        sub_delta=sub_bitstring_*multi
        sub_delta_A_sub=sub_delta.T@L@sub_delta
        ncut+=sub_delta_A_sub
    return ncut/2,best_bitstring

def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
        while nx.is_connected(nx_temp) is False:
            print('not full_connected')
            nx_temp = nx.fast_gnp_random_graph(n, p)
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
    elif graph_type == 'erdos':
        nx_temp = nx.erdos_renyi_graph(n=n, p=p, seed=random_seed)
        while nx.is_connected(nx_temp) is False:
            print('not full_connected')
            nx_temp = nx.erdos_renyi_graph(n=n, p=p)
        print(f'Generating erdos-renyi graph with n={n}, p={p}, seed={random_seed}')
    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph
