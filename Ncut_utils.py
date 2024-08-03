
import torch
import networkx as nx
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from utils.GenerateUniformBAgraphs import RandomBarabasiAlbertGraphGenerator,EdgeType
from itertools import chain, islice
from time import time
import numpy as np
from scipy.linalg import eigh

def normalization(adj_torch,degree_torch):
    for i in range(adj_torch.shape[0]):
        adj_torch[i,i]=1
        degree_torch[i,i]+=1
    degree_hat=torch.where(degree_torch!=0,degree_torch**(-0.5),degree_torch)
    return degree_hat@adj_torch@degree_hat

class GCN_dev(nn.Module):
    def __init__(self,in_feats,hidden_size,number_classes,dropout,device,heads):
        super(GCN_dev, self).__init__()
        self.gat_layers=nn.ModuleList()
        #two-layer GAT
        self.gat_layers.append(
            dglnn.GATConv(
                in_feats,
                hidden_size,
                heads[0],
                feat_drop=0.2,
                attn_drop=0.2,
                activation=F.elu,
            )
        )
        self.gat_layers.append(
            dglnn.GATConv(
                hidden_size*heads[0],
                # number_classes,
                hidden_size,
                heads[1],
                feat_drop=0.2,
                attn_drop=0.2,
                residual=True,
                activation=F.elu,
            )
        )
        self.gat_layers.append(dglnn.GATConv(
            hidden_size*heads[1],
            # number_classes,
            hidden_size,
            heads[2],
            feat_drop=0.2,
            attn_drop=0.2,
            residual=True,
            activation=F.elu,
        ))
        self.gat_layers.append(dglnn.GATConv(
            hidden_size*heads[2],
            number_classes,
            heads[3],
            residual=True,
            activation=F.softmax,
        ))
    def forward(self,g,inputs):
        h=inputs
        for i,layer in enumerate(self.gat_layers):
            h=layer(g,h)
            if i ==3:
                h=h.mean(1)
            else:
                h=h.flatten(1)
        return h


def generate_graph(n, d=None, p=None, graph_type='reg', random_seed=0):
    """
    Helper function to generate a NetworkX random graph of specified type,
    given specified parameters (e.g. d-regular, d=3). Must provide one of
    d or p, d with graph_type='reg', and p with graph_type in ['prob', 'erdos'].

    Input:
        n: Problem size
        d: [Optional] Degree of each node in graph
        p: [Optional] Probability of edge between two nodes
        graph_type: Specifies graph type to generate
        random_seed: Seed value for random generator
    Output:
        nx_graph: NetworkX OrderedGraph of specified type and parameters
    """
    if graph_type == 'reg':
        print(f'Generating d-regular graph with n={n}, d={d}, seed={random_seed}')
        nx_temp = nx.random_regular_graph(d=d, n=n, seed=random_seed)
    elif graph_type == 'prob':
        nx_temp = nx.fast_gnp_random_graph(n, p, seed=random_seed)
        while nx.is_connected(nx_temp) is False:
            print('not connected')
            nx_temp = nx.fast_gnp_random_graph(n, p)
        print(f'Generating p-probabilistic graph with n={n}, p={p}, seed={random_seed}')
    elif graph_type == 'erdos':
        nx_temp = nx.erdos_renyi_graph(n=n, p=p, seed=random_seed)
        while nx.is_connected(nx_temp) is False:
            print('not connected')
            nx_temp = nx.erdos_renyi_graph(n=n, p=p)
        print(f'Generating erdos-renyi graph with n={n}, p={p}')
    elif graph_type =='ba':
        print(f'Generating ba graph with n={n}, seed={random_seed}')
        test_graph_generater = RandomBarabasiAlbertGraphGenerator(n_spins=n, m_insertion_edges=4,
                                                                  edge_type=EdgeType.UNIFORM)
        generate_matrix = test_graph_generater.get(random_seed=random_seed)
        nx_temp = nx.from_numpy_array(generate_matrix)

    else:
        raise NotImplementedError(f'!! Graph type {graph_type} not handled !!')

    # Networkx does not enforce node order by default
    nx_temp = nx.relabel.convert_node_labels_to_integers(nx_temp)
    # Need to pull nx graph into OrderedGraph so training will work properly
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(sorted(nx_temp.nodes()))
    nx_graph.add_edges_from(nx_temp.edges)
    return nx_graph


# helper function to convert Q dictionary to torch tensor
def qubo_dict_to_torch(nx_G, Q, torch_dtype=None, torch_device=None):
    """
    Output Q matrix as torch tensor for given Q in dictionary format.

    Input:
        Q: QUBO matrix as defaultdict
        nx_G: graph as networkx object (needed for node lables can vary 0,1,... vs 1,2,... vs a,b,...)
    Output:
        Q: QUBO as torch tensor
    """

    # get number of nodes
    n_nodes = len(nx_G.nodes)

    # get QUBO Q as torch tensor
    Q_mat = torch.zeros(n_nodes, n_nodes)
    for (x_coord, y_coord), val in Q.items():
        Q_mat[x_coord][y_coord] = val

    if torch_dtype is not None:
        Q_mat = Q_mat.type(torch_dtype)

    if torch_device is not None:
        Q_mat = Q_mat.to(torch_device)

    return Q_mat

def gen_degree_torch(nx_G,adj_torch,torch_dtype=None,torch_device=None):
    n_nodes = len(nx_G.nodes)
    D_mat = torch.zeros(n_nodes, n_nodes)
    for i in range(n_nodes):
        degree = sum(adj_torch[i, :])
        D_mat[i, i] = degree
    if torch_dtype is not None:
        D_mat = D_mat.type(torch_dtype)
    if torch_device is not None:
        D_mat = D_mat.to(torch_device)
    return D_mat

# Chunk long list
def gen_combinations(combs, chunk_size):
    yield from iter(lambda: list(islice(combs, chunk_size)), [])


# helper function for custom loss according to Q matrix
def loss_func(probs_processed,degree_torch,adj_torch,probs_detach=None):
    """
    Function to compute cost value for given probability of spin [prob(+1)] and predefined Q matrix.

    Input:
        probs: Probability of each node belonging to each class, as a vector
        Q_mat: QUBO as torch tensor
    """

    loss=0
    if probs_detach!=None:
        for i in range(probs_processed.shape[1]):
            sub_detach = probs_detach[:, i]
            sub_detach=torch.unsqueeze(sub_detach,1)
            subprobs=torch.unsqueeze(probs_processed[:,i],1)
            L=degree_torch-adj_torch
            probsT_D_probs=((subprobs.T@degree_torch@subprobs)[0,0])**(-1/2)
            subprobs_delta=subprobs*probsT_D_probs
            single_trace=subprobs_delta.T@L@subprobs_delta
            loss+=single_trace
    else:
        for i in range(probs_processed.shape[1]):
            subprobs=torch.unsqueeze(probs_processed[:,i],1)
            L=degree_torch-adj_torch
            probsT_D_probs=((subprobs.T@degree_torch@subprobs)[0,0])**(-1/2)
            subprobs_delta=subprobs*probsT_D_probs
            single_trace = subprobs_delta.T @ L @ subprobs_delta
            loss+=single_trace
    return (1/probs_processed.shape[1])*loss




# Construct graph to learn on
def get_gnn(degree_torch,adj_torch,n_nodes, gnn_hypers, opt_params, torch_device, torch_dtype):
    """
    Generate GNN instance with specified structure. Creates GNN, retrieves embedding layer,
    and instantiates ADAM optimizer given those.

    Input:
        n_nodes: Problem size (number of nodes in graph)
        gnn_hypers: Hyperparameters relevant to GNN structure
        opt_params: Hyperparameters relevant to ADAM optimizer
        torch_device: Whether to load pytorch variables onto CPU or GPU
        torch_dtype: Datatype to use for pytorch variables
    Output:
        net: GNN instance
        embed: Embedding layer to use as input to GNN
        optimizer: ADAM optimizer instance
    """
    dim_embedding = gnn_hypers['dim_embedding']
    hidden_dim = gnn_hypers['hidden_dim']
    dropout = gnn_hypers['dropout']
    number_classes = gnn_hypers['number_classes']

    # instantiate the GNN
    # net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device,[4,4,8,8])#0.9900
    # net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device,[4,4,8,12])#0.9888
    # net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device,[4,4,12,12])#0.9890
    net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device,[4,8,8,12])#0.9877（2000 erdosit）
    # net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device,[8,8,8,12])#0.9936
    # net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device,[8,8,12,16])#0.9936
    # net = GCN_dev(dim_embedding, hidden_dim, number_classes, dropout, torch_device, [4, 4, 8, 8,12])

    net = net.type(torch_dtype).to(torch_device)

    degree_torch=degree_torch.cpu().numpy()
    adj_torch=adj_torch.cpu().numpy()
    L=degree_torch-adj_torch
    eigenvalues, eigenvectors = eigh(L,degree_torch)
    tensor=torch.FloatTensor(eigenvectors)
    embed=nn.Embedding.from_pretrained(tensor)
    embed = nn.Embedding(n_nodes, dim_embedding)

    embed = embed.type(torch_dtype).to(torch_device)

    # set up Adam optimizer
    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, **opt_params)
    return net, embed, optimizer


# Parent function to run GNN training given input config
def run_gnn_training(number_classes,degree_torch,adj_torch, dgl_graph, net, embed, optimizer, number_epochs, tol, patience, prob_threshold,device,type):
    """
    Wrapper function to run and monitor GNN training. Includes early stopping.
    """
    # Assign variable for user reference
    inputs = embed.weight
    # inputs=F.normalize(inputs,p=2,dim=1)

    prev_loss = 1.  # initial loss value (arbitrary)
    count = 0       # track number times early stopping is triggered
    change_patience=0

    # initialize optimal solution
    best_bitstring = torch.ones((dgl_graph.number_of_nodes(),number_classes)).type(type).to(device)
    for i in range(dgl_graph.number_of_nodes()):
        best_bitstring[i,best_bitstring[i,:].argmax()]=1
        for j in range(number_classes):
            if j!=best_bitstring[i,:].argmax():
                best_bitstring[i,j]=0
    best_loss = loss_func(best_bitstring,degree_torch,adj_torch)
    print(best_loss)
    best_epoch=-1
    best_Ncut=loss_func(best_bitstring,degree_torch,adj_torch)
    # print('best_cut',best_Ncut)
    t_gnn_start = time()

    # Training logic
    for epoch in range(number_epochs):
        # normalized_adj=normalization(adj_torch,degree_torch)
        # get logits/activations
        probs = net(dgl_graph, inputs.data)  # collapse extra dimension output from model
        probs_detach=probs.clone()
        for i in range(probs_detach.shape[0]):
            probs_detach[i,probs_detach[i,:].argmax()]=1
            for j in range(probs_detach.shape[1]):
                if j!=probs_detach[i,:].argmax():
                    probs_detach[i,j]=0
        # build cost value with QUBO cost function
        loss = loss_func(probs,degree_torch,adj_torch,probs_detach)
        loss_ = loss.detach().item()
        bitstring=probs_detach
        true_ratio_cut=loss_func(probs_detach,degree_torch,adj_torch)
        # Apply projection
        best_Ncut=best_Ncut.detach()
        if  true_ratio_cut<best_Ncut or np.isnan(best_Ncut.cpu().numpy()):
            best_loss = loss
            best_bitstring = bitstring
            best_epoch=epoch
            best_Ncut=true_ratio_cut
            change_patience=0
        else:change_patience+=1

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss_}')

        # early stopping check
        # If loss increases or change in loss is too small, trigger
        if (abs(loss_ - prev_loss) <= tol) | ((loss_ - prev_loss) > 0):
            count += 1
        else:
            count = 0

        if count >= patience:
            print(f'Stopping early on epoch {epoch} (patience: {patience})')
            break
        if change_patience>=500:
            break
        # update loss tracking
        prev_loss = loss_

        # run optimization with backpropagation
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()        # calculate gradient through compute graph
        optimizer.step()       # take step, update weights

    t_gnn = time() - t_gnn_start
    print(f'GNN training (n={dgl_graph.number_of_nodes()}) took {round(t_gnn, 3)}')
    return net, epoch, best_bitstring,best_epoch,best_Ncut