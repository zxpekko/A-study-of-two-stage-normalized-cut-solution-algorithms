import time
from copy import deepcopy

import dgl
import networkx as nx
import numpy as np
import torch
import torch.nn as nn

from utils.GenerateUniformBAgraphs import RandomBarabasiAlbertGraphGenerator, EdgeType
from utils.solve_max_cut import postprocess_gnn_max_cut, solve_normalized_cut, postprocess_normalzed_cut, \
    generate_graph

VERY_LARGE_INT = 10  # 65536
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TORCH_DTYPE = torch.float64


class Env(object):
    def __init__(self, solver_type='g', node=80, graph_type='er', test=False, low_stage=None):
        self.solver_type = solver_type
        self.available_solvers = ('g')
        self.n_samples = 50
        self.n_samples_test = 20
        self.train_samples = []
        self.test_samples = []
        # number of the node and type of the graph
        self.test = test
        self.node = node
        self.graph_type = graph_type

        if low_stage == 'gnn':
            self.gnn_model = torch.load(f'gnn_models/gnn_model_{self.graph_type}_node_{node}')
            assert self.gnn_model is not None, "load gnn_model_failed"
        elif low_stage == 'spc':
            self.gnn_model = None

        self.embed = nn.Embedding(self.node, self.node).type(TORCH_DTYPE).to(TORCH_DEVICE)
        self.low_stage_time = 0
        self.low_stage = low_stage
        assert low_stage is not None, "low stage error"
        self.process_dataset()
        assert solver_type in self.available_solvers

    def process_dataset(self):
        if self.graph_type == 'er':
            print('er')
            for i in range(self.n_samples):
                g = generate_graph(n=self.node, p=0.15, graph_type='erdos', random_seed=i)
                self.train_samples.append(g)

            for i in range(self.n_samples_test):
                if self.test:
                    g = generate_graph(n=self.node, p=0.15, graph_type='erdos', random_seed=-10 * i)
                else:
                    g = generate_graph(n=self.node, p=0.15, graph_type='erdos', random_seed=-100 * i)
                self.test_samples.append(g)

        elif self.graph_type == 'prob':
            print('prob')
            for i in range(self.n_samples):
                g = generate_graph(n=self.node, p=0.15, graph_type='prob', random_seed=i)

                self.train_samples.append(g)

            for i in range(self.n_samples_test):

                if self.test:
                    g = generate_graph(n=self.node, p=0.15, graph_type='prob', random_seed=-10 * i)
                else:
                    g = generate_graph(n=self.node, p=0.15, graph_type='prob', random_seed=-100 * i)
                self.test_samples.append(g)


        elif self.graph_type == 'reg':
            print('reg')
            for i in range(self.n_samples):
                g = generate_graph(n=self.node, d=5, p=0.15, graph_type='reg', random_seed=i)

                self.train_samples.append(g)

            for i in range(self.n_samples_test):
                if self.test:
                    g = generate_graph(n=self.node, d=5, p=0.15, graph_type='reg', random_seed=-10 * i)
                else:
                    g = generate_graph(n=self.node, d=5, p=0.15, graph_type='reg', random_seed=-100 * i)
                self.test_samples.append(g)

        elif self.graph_type == 'ba':
            print('ba')
            test_graph_generater = RandomBarabasiAlbertGraphGenerator(n_spins=self.node, m_insertion_edges=4,
                                                                      edge_type=EdgeType.UNIFORM)
            for i in range(self.n_samples):
                generate_matrix = test_graph_generater.get(random_seed=i)
                g = nx.from_numpy_array(generate_matrix)
                self.train_samples.append(g)
            for i in range(self.n_samples_test):
                if self.test:
                    generate_matrix = test_graph_generater.get(random_seed=-10 * i)
                else:
                    generate_matrix = test_graph_generater.get(random_seed=-100 * i)
                g = nx.from_numpy_array(generate_matrix)
                self.test_samples.append(g)

    def generate_tuples(self):
        training_tuples = []
        testing_tuples = []
        for i in range(self.n_samples):
            g = self.train_samples[i]
            if self.low_stage == 'gnn':

                graph_dgl = dgl.from_networkx(nx_graph=g)
                graph_dgl = dgl.add_self_loop(graph_dgl)
                graph_dgl = graph_dgl.to(TORCH_DEVICE)
                probs = self.gnn_model(graph_dgl, self.embed.weight)
                probs_detach = probs.clone()
                for i in range(probs_detach.shape[0]):
                    probs_detach[i, probs_detach[i, :].argmax()] = 1
                    for j in range(probs_detach.shape[1]):
                        if j != probs_detach[i, :].argmax():
                            probs_detach[i, j] = 0
                new_tour = probs_detach
                new_tour = new_tour.cpu().detach()
            elif self.low_stage == 'spc':
                new_tour = solve_normalized_cut(g)
            new_solution, _ = postprocess_normalzed_cut(new_tour, g)
            new_solution = new_solution[0][0]

            maxcut_solutions = {}
            maxcut_solutions['g'] = new_solution
            A = np.array(nx.adjacency_matrix(g).todense())
            all_edges = g.edges()
            num_nodes = g.number_of_nodes()
            edge_candidates = {x: set() for x in range(num_nodes)}
            for edge_pair in all_edges:
                a, b = edge_pair[0], edge_pair[1]
                edge_candidates[a].add(b)
                edge_candidates[b].add(a)

            training_tuples.append((
                A,  # lower-left triangle of adjacency matrix
                edge_candidates,  # edge candidates
                maxcut_solutions['g']
            ))
        start_time = time.time()
        for i in range(self.n_samples_test):
            g = self.test_samples[i]
            if self.low_stage == 'gnn':

                graph_dgl = dgl.from_networkx(nx_graph=g)
                graph_dgl = dgl.add_self_loop(graph_dgl)
                graph_dgl = graph_dgl.to(TORCH_DEVICE)
                probs = self.gnn_model(graph_dgl, self.embed.weight)
                probs_detach = probs.clone()
                for i in range(probs_detach.shape[0]):
                    probs_detach[i, probs_detach[i, :].argmax()] = 1
                    for j in range(probs_detach.shape[1]):
                        if j != probs_detach[i, :].argmax():
                            probs_detach[i, j] = 0
                new_tour = probs_detach
                new_tour = new_tour.cpu().detach()
            elif self.low_stage == 'spc':
                new_tour = solve_normalized_cut(g)

            new_solution, _ = postprocess_normalzed_cut(new_tour, g)
            new_solution = new_solution[0][0]
            maxcut_solutions = {}
            maxcut_solutions['g'] = new_solution
            A = np.array(nx.adjacency_matrix(g).todense())
            all_edges = g.edges()
            num_nodes = g.number_of_nodes()
            edge_candidates = {x: set() for x in range(num_nodes)}
            for edge_pair in all_edges:
                a, b = edge_pair[0], edge_pair[1]
                edge_candidates[a].add(b)
                edge_candidates[b].add(a)
            testing_tuples.append((
                A,  # lower-left triangle of adjacency matrix
                edge_candidates,  # edge candidates
                maxcut_solutions['g']
            ))

        self.low_stage_time = time.time() - start_time
        return training_tuples, testing_tuples

    def evaluate(self, g, sol):
        size_max_cut, cut_value, cut_solution = postprocess_gnn_max_cut(sol, g)
        return cut_value

    def step(self, graph_index, list_lower_matrix, act, prev_solution, test):
        g = nx.from_numpy_array(list_lower_matrix)
        if test:
            ori_graph = self.test_samples[graph_index]
        else:
            ori_graph = self.train_samples[graph_index]
        new_list_lower_matrix = deepcopy(list_lower_matrix)
        if isinstance(act, torch.Tensor):
            act = (act[0].item(), act[1].item())
        if act[0] >= act[1]:
            idx0, idx1 = act[0], act[1]
        else:
            idx0, idx1 = act[1], act[0]
        new_list_lower_matrix[idx0][idx1] += VERY_LARGE_INT
        new_list_lower_matrix[idx1][idx0] += VERY_LARGE_INT
        new_g = nx.from_numpy_array(new_list_lower_matrix)
        num_nodes = new_g.number_of_nodes()
        if self.low_stage == 'gnn':
            graph_dgl = dgl.from_networkx(nx_graph=g)
            graph_dgl = dgl.add_self_loop(graph_dgl)
            graph_dgl = graph_dgl.to(TORCH_DEVICE)
            probs = self.gnn_model(graph_dgl, self.embed.weight)
            probs_detach = probs.clone()
            for i in range(probs_detach.shape[0]):
                probs_detach[i, probs_detach[i, :].argmax()] = 1
                for j in range(probs_detach.shape[1]):
                    if j != probs_detach[i, :].argmax():
                        probs_detach[i, j] = 0
            new_tour = probs_detach
            new_tour = new_tour.cpu().detach()
        elif self.low_stage == 'spc':
            new_tour = solve_normalized_cut(g)

        new_solution, _ = postprocess_normalzed_cut(new_tour, ori_graph)
        all_new_edges = new_g.edges()
        new_edge_candidate = {x: set() for x in range(num_nodes)}
        for edge_pair in all_new_edges:
            a, b = edge_pair[0], edge_pair[1]
            new_edge_candidate[a].add(b)
            new_edge_candidate[b].add(a)
        reward = prev_solution - new_solution
        done = False
        return reward, new_list_lower_matrix, new_edge_candidate, new_solution, done

    def step_e2e(self, list_lower_matrix, prev_act, act, prev_solution):
        new_list_lower_matrix = deepcopy(list_lower_matrix)
        if isinstance(prev_act, torch.Tensor):
            prev_act = prev_act.item()
        if isinstance(act, torch.Tensor):
            act = act.item()
        if prev_act is not None:
            if prev_act > act:
                new_list_lower_matrix[prev_act][act] += VERY_LARGE_INT
                step_cost = list_lower_matrix[prev_act][act]
            else:
                new_list_lower_matrix[act][prev_act] += VERY_LARGE_INT
                step_cost = list_lower_matrix[act][prev_act]
        else:
            step_cost = 0
        new_solution = prev_solution + step_cost
        node_candidates = self.get_node_candidates(new_list_lower_matrix, len(new_list_lower_matrix))
        if len(node_candidates) == 0:
            done = True
            last_act1, last_act2 = self.get_node_candidates(new_list_lower_matrix, len(new_list_lower_matrix),
                                                            last_act=True)
            if last_act1 > last_act2:
                last_step_cost = list_lower_matrix[last_act1][last_act2]
            else:
                last_step_cost = list_lower_matrix[last_act2][last_act1]
            new_solution = new_solution + last_step_cost
        else:
            done = False
        return prev_solution - new_solution, new_list_lower_matrix, node_candidates, new_solution, done

    @staticmethod
    def edge_candidate_from_tour(tour, num_nodes):
        print('tour', tour)
        assert tour[0] == tour[-1], '不能形成回路'
        edge_candidate = {x: set() for x in range(num_nodes)}
        iter_obj = iter(tour)
        last_node = next(iter_obj)
        for node in iter_obj:
            edge_candidate[last_node].add(node)
            edge_candidate[node].add(last_node)
            last_node = node
        return edge_candidate

    @staticmethod
    def get_node_candidates(list_lower_mat, num_nodes, last_act=False):
        visited_once = set()
        visited_twice = set()
        for i in range(num_nodes):
            for j in range(i + 1):
                if i != j:
                    if list_lower_mat[i][j] > VERY_LARGE_INT:
                        if i in visited_once:
                            visited_twice.add(i)
                            visited_once.remove(i)
                        else:
                            visited_once.add(i)
                        if j in visited_once:
                            visited_twice.add(j)
                            visited_once.remove(j)
                        else:
                            visited_once.add(j)
        if last_act:
            assert len(visited_once) == 2
            return visited_once
        else:
            candidates = list(range(num_nodes))
            for i in visited_twice:
                candidates.remove(i)
            for i in visited_once:
                candidates.remove(i)
            return candidates

if __name__ == '__main__':
    env = Env()
    env.step()
