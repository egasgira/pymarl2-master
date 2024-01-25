from collections import deque
import networkx as nx
import numpy as np
import random
import torch


# TODO: Put in a class Matrix.
def generate_matrix(graph, safe_nodes, danger_nodes):
    nodes = sorted(graph.nodes)
    n_nodes = len(nodes)

    # Matrix has 3 channels: Distance, Bandwidth, Number of people in nodes and adges
    # People: Diagonal is the number of persons in nodes and the rest n persons in edges
    matrix = np.zeros((n_nodes, n_nodes, 3))

    # People in origin node
    n_origin_nodes = np.random.randint(1, int(n_nodes / 2))
    origin_nodes = sorted(random.sample(range(n_nodes), n_origin_nodes))
    max_persons = 10

    for i in origin_nodes:
        matrix[i, i, 2] = np.random.randint(1, max_persons)

    # Get the matrix of path distances
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            # Distance matrix
            edge = graph.get_edge_data(node_i, node_j)
            if edge:
                # For now we just get the first edge between nodes (index 0)
                # there could be multiple edges between nodes, differents ways, street, highway...
                matrix[i][j][0] = edge[0]['length']

                # TODO: what if multiple edges (street, highway...) exist between nodes, now only one is taken into consideration
                if i != j:
                    # BW only in edges. Check if value for this edge has already been setted
                    matrix[i][j][1] = matrix[j][i][1] if matrix[j][i][1] > 0\
                        else np.random.randint(5, int(max_persons * 0.75))

    for safe_node in safe_nodes:
        matrix[safe_node][safe_node][2] = 0

    # for danger_node in danger_nodes:  # TODO remove this part, people should be able to start from this point and escape later
    #     matrix[danger_node][danger_node][2] = 0

    return matrix


def display_matrix():
    pass


def _get_min_distances_to_safe_nodes(G, safe_nodes):
    nodes = sorted(G.nodes)
    safe_node_ids = [nodes[safe_node]
                     for safe_node in safe_nodes]

    min_distances_to_safe_node = []
    for node_id in nodes:
        distances = []
        for safe_node_id in safe_node_ids:
            distance = nx.shortest_path_length(
                G, source=node_id, target=safe_node_id, weight="length")
            distances.append(distance)
        min_distances_to_safe_node.append(min(distances))

    return min_distances_to_safe_node


def _get_min_distances_to_danger_nodes(G, danger_nodes):
    nodes = sorted(G.nodes)
    danger_node_ids = [nodes[danger_node]
                       for danger_node in danger_nodes]

    min_distances_to_danger_node = []
    for node_id in nodes:
        distances = []
        for danger_node_id in danger_node_ids:
            distance = nx.shortest_path_length(
                G, source=node_id, target=danger_node_id, weight="length")
            distances.append(distance)
        min_distances_to_danger_node.append(min(distances))

    return min_distances_to_danger_node


def _clean_data(G):
    # Remove isolated nodes from the graph
    isolated_nodes = [node for node in G.nodes() if not list(
        G.neighbors(node))] + list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)

    return G


class ReplayMemory:
    """This class allows to store transitions between states (state, reward, next_state).

    Attributes:
        memory: List where tuples (state, action, reward, next_state) are stored.

    Methods:
        push: Add tuple to the end of the memory list.
        sample: Get random tuple sample from memory.
        len: Get buffer length.
    """

    def __init__(self, n_nodes, maxlen=500):
        self.memory = deque(maxlen=maxlen)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.n_nodes = n_nodes

    def push(self, state, action, reward, next_state, done):
        """Push adds a tuple to the end of the memory list.
        """
        experience = (state, action, reward, next_state, done)
        self.memory.append(experience)

    def sample(self, num_samples):
        state_size = (self.n_nodes, self.n_nodes, 3)
        action_size = (self.n_nodes, self.n_nodes)

        states = np.empty((num_samples,) + state_size)
        actions = np.empty((num_samples,) + action_size)
        rewards = np.empty(num_samples)
        next_states = np.empty((num_samples,) + state_size)
        dones = np.empty(num_samples)

        idx = np.random.choice(len(self.memory), num_samples)
        for i, buffer_idx in enumerate(idx):
            state, action, reward, next_state, done = self.memory[buffer_idx]
            states[i] = state
            actions[i] = action
            rewards[i] = reward
            next_states[i] = next_state
            dones[i] = done

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)