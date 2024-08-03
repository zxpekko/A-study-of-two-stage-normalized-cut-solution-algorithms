import dgl
import torch
import random
import os
import numpy as np
from collections import defaultdict
from time import time

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

seed_value = 1
random.seed(seed_value)        # seed python RNG
np.random.seed(seed_value)     # seed global NumPy RNG
torch.manual_seed(seed_value)  # seed torch RNG
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float64
print(f'Will use device: {TORCH_DEVICE}, torch dtype: {TORCH_DTYPE}')
from  Ncut_utils import generate_graph, get_gnn, run_gnn_training, qubo_dict_to_torch, gen_combinations, loss_func,gen_degree_torch
kmax=100
lmax=200
max_iteration=10

def gen_q_dict_cut(nx_G,localdegree=2):
    Q_dic = defaultdict(int)
    for (u, v) in nx_G.edges:
        hasweight = len(list(nx_G.get_edge_data(u, v).values()))
        break
    if (hasweight == 0):
        for (u, v) in nx_G.edges:
            Q_dic[(u, v)] = -localdegree
        for u in nx_G.nodes:
            Q_dic[(u, u)] = nx_G.degree(u)
    else:
        for (u, v) in nx_G.edges:
            Q_dic[(u, v)] = -localdegree * list(nx_G.get_edge_data(u, v).values())[0]
        for u in nx_G.nodes:
            diag_weight = 0
            for (u1, v1) in nx_G.edges:
                if u == u1 or u == v1:
                    diag_weight += list(nx_G.get_edge_data(u1, v1).values())[0]
            Q_dic[(u, u)] = diag_weight
    return Q_dic

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

def k_Ncut(bitstring,q_torch,degree_torch):
    Ncut=0
    for i in range(bitstring.shape[1]):
        sub_bitstring=bitstring[:,i]
        sub_bitstring=torch.unsqueeze(sub_bitstring,1)
        cut_vaue=sub_bitstring.T@q_torch@sub_bitstring
        degree=sub_bitstring.T@degree_torch@sub_bitstring
        single_Ncut=cut_vaue/degree
        Ncut+=single_Ncut
    return (1/bitstring.shape[1])*Ncut

def shaking(best_bitstring):
    shanking_array = np.random.permutation(best_bitstring.shape[0])
    j=0
    best_bitstring_clone=best_bitstring.clone()
    while j<best_bitstring_clone.shape[0]-1:
        best_bitstring_clone[[shanking_array[j],shanking_array[j+1]],:]=best_bitstring_clone[[shanking_array[j+1],shanking_array[j]],:]
        j+=2
    return best_bitstring_clone

def Neighborhood_action(bitstring):
    local_list=[]
    for l in range(lmax):
        bitstringclone = bitstring.clone()
        for i in range(bitstringclone.shape[0]):
            j=0
            while bitstringclone[i,j]==0:
                j+=1
            bitstringclone[i,j]=0
            randnumber=random.randrange(0,bitstring.shape[1])
            bitstringclone[i,randnumber]=1
        local_list.append(bitstringclone)
    return local_list

def VND(q_torch,degree_torch,best_Ncut,best_bitstring):
    vns_patience=100
    count=0
    origin_bitstring=best_bitstring
    local_list=Neighborhood_action(best_bitstring)
    for i in range(len(local_list)):
        k_ncut=k_Ncut(local_list[i],q_torch,degree_torch)
        if k_ncut<best_Ncut:
            best_Ncut=k_ncut
            best_bitstring=local_list[i]
    for iteration in range(max_iteration):
        print(f'epoch{iteration}...')
        shaking_bitstring=shaking(origin_bitstring)
        local_list=Neighborhood_action(shaking_bitstring)
        local_best_cut=1e5
        local_best_bitstring=None
        for i in range(len(local_list)):
            k_ncut=k_Ncut(local_list[i],q_torch,degree_torch)
            if k_ncut<local_best_cut:
                local_best_cut=k_ncut
                local_best_bitstring=local_list[i]
        if local_best_cut<best_Ncut:
            best_Ncut=local_best_cut
            best_bitstring=local_best_bitstring
            print('change')
            print('current best ncut',best_Ncut)
            count=0
        else:count+=1
        if count>=vns_patience:
            break
    return best_bitstring

# Graph hypers/
n = 40
d = 5
p = 0.15
graph_type = 'erdos'

# NN learning hypers #
number_epochs = int(1e4)
learning_rate = 2e-3
PROB_THRESHOLD = 0.5
k=2
# Early stopping to allow NN to train to near-completion
tol = 1e-4          # loss must change by more than tol, or trigger
patience = number_epochs   # number early stopping triggers before breaking loop

# dim_embedding = int(np.sqrt(n))    # e.g. 10
dim_embedding = int(n)
# print(dim_embedding)

# nx_graph = generate_graph(n=n, d=d, p=p, graph_type=graph_type, random_seed=seed_value)
Ncut_list=[]
count=0
graph_nums=1

for i in range(graph_nums):
    nx_graph = generate_graph(n=n, d=d, p=p, graph_type=graph_type)
    n=nx_graph.number_of_nodes()
    hidden_dim = int(dim_embedding/2)
    graph_dgl = dgl.from_networkx(nx_graph=nx_graph)
    graph_dgl = graph_dgl.to(TORCH_DEVICE)
    q_torch=qubo_dict_to_torch(nx_graph,gen_q_dict_cut(nx_graph),torch_dtype=TORCH_DTYPE,torch_device=TORCH_DEVICE)
    adj_torch=qubo_dict_to_torch(nx_graph,gen_adj_dict(nx_graph),torch_dtype=TORCH_DTYPE,torch_device=TORCH_DEVICE)
    for i in range(adj_torch.shape[0]):
        adj_torch[i,i]=1
    degree_torch=gen_degree_torch(nx_graph,adj_torch,torch_dtype=TORCH_DTYPE,torch_device=TORCH_DEVICE)
    opt_params = {'lr': learning_rate}
    gnn_hypers = {
        'dim_embedding': dim_embedding,
        'hidden_dim': hidden_dim,
        'dropout': 0.0,
        'number_classes': k,
        'prob_threshold': PROB_THRESHOLD,
        'number_epochs': number_epochs,
        'tolerance': tol,
        'patience': patience
    }

    net, embed, optimizer = get_gnn(degree_torch,adj_torch,n, gnn_hypers, opt_params, TORCH_DEVICE, TORCH_DTYPE)

    # For tracking hyperparameters in results object
    gnn_hypers.update(opt_params)
    print('Running GNN...')
    gnn_start = time()
    net, epoch, best_bitstring,best_epoch,best_Ncut= run_gnn_training(gnn_hypers['number_classes'],
        degree_torch,adj_torch, graph_dgl, net, embed, optimizer, gnn_hypers['number_epochs'],
        gnn_hypers['tolerance'], gnn_hypers['patience'], gnn_hypers['prob_threshold'],TORCH_DEVICE,TORCH_DTYPE)

    if graph_type == 'erdos':
        graph_type = 'er'
    torch.save(net,f'gnn_models/gnn_model_{graph_type}_node_{n}')
    gnn_time = time() - gnn_start
    for i in range(gnn_hypers['number_classes']):
        print(sum(best_bitstring[:,i]))
    print('best_Ncut',best_Ncut)
    print('best_epoch',best_epoch)
    Ncut=0
    cut_value_list=[]
    degree_list=[]
    for i in range(gnn_hypers['number_classes']):
        sub_bitstring=best_bitstring[:,i]
        sub_bitstring=torch.unsqueeze(sub_bitstring,1)
        cut_vaue=sub_bitstring.T@q_torch@sub_bitstring
        cut_value_list.append(cut_vaue)
        degree=sub_bitstring.T@degree_torch@sub_bitstring
        degree_list.append(degree)
        single_Ncut=cut_vaue/degree
        Ncut+=single_Ncut
    true_k_Ncut=(1/gnn_hypers['number_classes'])*Ncut
    Ncut_list.append(true_k_Ncut)
    count+=1
print('mean normalized cut',sum(Ncut_list)/len(Ncut_list))