import random
import numpy as np
import math
from envs.multiagentenv import MultiAgentEnv

import numpy as np
import os
import osmnx as ox
import random

from envs.CityNavigation.utils import generate_matrix, _get_min_distances_to_safe_nodes, _get_min_distances_to_danger_nodes, _clean_data


class streetmap_multiagent(MultiAgentEnv):

    def __init__(self, map_name='very_small_map', safe_nodes=None, danger_nodes=None, speed=1.56, episode_limit =100, n_persons = 2, individual_reward = False, **kwargs):

        # safe_nodes is a list of indices
        self.terminations = {}
        self.truncated = {}
        self.episode_limit = episode_limit
        if safe_nodes is None:
            # safe_nodes = sorted(random.sample(range(n_nodes), 2))
            safe_nodes = [8]

        if danger_nodes is None:
            # danger_nodes = sorted(random.sample(range(n_nodes), 1))
            danger_nodes = []

        # self.graph = graph
        # self.nodes = sorted(graph.nodes)
        self._initialize_matrix(map_name)
        self.n_nodes = self.matrix.shape[0]
        self.min_distances_to_safe_node = self.min_distances_to_safe_node
        self.min_distances_to_danger_node = self.min_distances_to_danger_node

        # Initial state
        self.state = np.copy(self.matrix) # a 3d matrix showing, the band widt of the nodes, the lenght to the nearest nodes and the number of agents in each node.
        self.initial_state = np.copy(self.matrix)

        for person in range(n_persons):
            index = random.randint(0, self.n_nodes-1)
            self.matrix[index, index, 2] += 1

        self.max_steps = episode_limit
        self.current_step = 0
        self.safe_nodes = safe_nodes
        self.danger_nodes = danger_nodes
        self.edge_data = [] # fjern
        #self.state_size = self.n_nodes# test2 * self.n_nodes * 3 + self.n_nodes
        self.state_size = self.n_nodes * self.n_nodes * 3 + self.n_nodes
        self.action_masks = {}

        self.observations = {} # An array of arrays with [state, position] where position is the node number of the agent, while the index is the agent number.
        self.rewards = {}
        self.next_state = None
        self.individual_reward = individual_reward

        # Movement speed
        self.speed = speed

        self.n_persons = np.sum(self.state[:, :, 2])
        self.max_dist = self.state[:, :, 0].max()
        self.max_band_widt = np.max(self.state[:, :, 1])

        #self.observation_space_length = self.n_nodes#**2 * 3 + self.n_nodes #test2
        self.observation_space_length = self.n_nodes**2 * 3 + self.n_nodes
        self.n_actions = self.n_nodes + 1
        self.agents = []
        self.dones = {}
        self.busy = {}
        self.rewards = {}

        # agents
        self.n_agents = n_persons

        # Statistics
        self.avg_timesteps_used = []
        self.avg_saved_persons = []

        # add a dicionary of agents with names equal to index. Types of agents can have different names in the future
        for i in range(self.n_agents):
            #self.agents.append(str(i))
            self.agents.append(i)
            self.dones[str(i)] = False
            self.busy[str(i)] = False
            self.rewards[str(i)] = 0

    def reset(self):
        observations = {}
        self.timestep = 0
        self.state = np.copy(self.initial_state)
        for person in range(self.n_agents):
            while True:
                index = random.randint(0, self.n_nodes - 1)
                if index not in self.safe_nodes:
                    break
            self.state[index, index, 2] += 1
        self._initialize_observations()

        # self.edge_data = []

        self.current_step = 0
        self.edge_data = []
        self._update_observations()



        for agent in self.agents:

            self.terminations[agent] = False
            self.truncated[agent] = False
            self.busy[agent] = False
            self._update_masks(agent)  # need to take the queue into account
            position_vector = np.zeros(self.n_nodes)
            position_vector[self.observations[agent][1]] = 1

            observations[agent] = {"observation": np.append(self.observations[agent][0].flatten(),
                                                            position_vector), #To know its own position
                                   "action_mask": self.action_masks[agent]}
        return self.get_obs(), self.get_state()

    def step(self, actions):
        observations = {}
        rewards = {}
        info = {}
        self.next_state = np.copy(self.state)

            # Perform action and update the environment state
        for agent in self.agents:
            if not self.terminations[agent] and self.current_step >= self.max_steps - 1:
                self.truncated[agent] = True
                self.terminations[agent] = True
                info["episode_limit"] = True 

            if not self.busy[agent] and not self.terminations[agent] and not self.truncated[agent] and actions[agent] != self.n_actions - 1: #new
                action_nr = actions[agent].copy()
                action = np.zeros(self.n_actions)
                action[action_nr] = 1
                if self.action_masks[agent][action_nr] != True:
                    action[action_nr] = False
                    action[self.observations[agent][1]] = True
                self._process_action(action, agent)


                temp_state = np.copy(self.next_state)
                temp_state[:,:,2] = 0
                temp_state[np.argmax(actions[agent]), np.argmax(actions[agent]), 2] = 1 #This is not totally correct, the agent get's reward just for doing the right action and not look at the state, but it should be the same just for test
                rewards[agent] = self._compute_distance_reward(temp_state)

        if self.current_step < self.max_steps: # -1?
            self._run_time_step()

        self._update_observations()
        self._update_masks()

        # Increment the step counter
        self.current_step += 1

        for agent in self.agents:
            if self.individual_reward:
                rewards[agent] = -self.min_distances_to_safe_node[np.argmax(actions[agent])]

            else:
                rewards[agent] = self._compute_distance_reward(self.state)
            position_vector = np.zeros(self.n_nodes)
            position_vector[self.observations[agent][1]] = 1
            observations[agent] = {"observation": np.append(self.observations[agent][0].flatten(),
                                                           position_vector),
                                  "action_mask": self.action_masks[agent]}
        termination = not False in self.terminations.values()
        if termination:
            self.avg_timesteps_used.append(self.current_step)
            self.avg_saved_persons.append(self.n_persons - np.sum(self.state[:, :, 2]))
        return rewards[0], termination, info#, self.truncated, self.busy

    def get_obs(self):
        observations = []
        for agent in self.agents:
            position_vector = np.zeros(self.n_nodes)
            position_vector[self.observations[agent][1]] = 1
            observations.append(np.append(self.observations[agent][0].flatten(),
                                                           position_vector))
        return observations

    def get_state(self):
        return self.state.flatten()

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]
    
    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.get_state_size() + self.n_nodes
    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        avl_actions = np.zeros(self.n_actions)
        avl_actions[self.action_masks[agent_id]] = 1
        return avl_actions
    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions
    def print_stats(self):
                stats = {
            "persons_saved": np.mean(self.avg_saved_persons[-100:]), 
            "time_steps": np.mean(self.avg_timesteps_used[-100:])
                }
                print(stats)
    
    def get_stats(self):
        stats = {
            "persons_saved": np.mean(self.avg_saved_persons[-100:]), 
            "time_steps": np.mean(self.avg_timesteps_used[-100:])
        }
        return stats

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError
    
    def _compute_distance_reward(self, next_state):
        indices = np.where(next_state[:, :, 2] > 0)

        reward = 0
        for i, j in zip(*indices):
            min_distance_to_safe_node = self.min_distances_to_safe_node[j] #j before
            reward -= min_distance_to_safe_node * next_state[i, j, 2]
        return reward

    def _compute_simple_reward(self, next_state):
        reward = np.sum(next_state[:, :, 2]) * -1
        return reward

    def _compute_danger_distance_reward(self, next_state):
        indices = np.where(next_state[:, :, 2] > 0)

        reward = 0
        for i, j in zip(*indices):
            min_distance_to_danger_node = self.min_distances_to_danger_node[j]
            # reward -= (min_distance_to_danger_node *
            #            next_state[i, j, 2]) ** self.current_step
            reward -= (min_distance_to_danger_node *
                       next_state[i, j, 2])

            reward += 10 * \
                      (np.sum(self.state[:, :, 2]) - np.sum(next_state[:, :, 2]))
        return reward

    def _run_time_step(self):
        # Iterate edge_data and update time_steps and node/edge state
        # element = [(origin_node, dst_node), time_step, agent_name]
        for elm_i, element in enumerate(self.edge_data):
            element[1] += 1

            i = element[0][0]
            j = element[0][1]

            # 1.56 (avg walking speed). 1 time step = 1 s
            required_time_steps = math.ceil(
                self.next_state[i, j, 0] / self.speed)
            # Min of 2 time steps per edge
            required_time_steps = max(required_time_steps, 0) #2

            if element[1] >= required_time_steps:
                self.next_state[i, j, 2] -= 1
                self.next_state[j, j, 2] += 1
                self.observations[element[2]][1] = j
                if j in self.safe_nodes:
                    self.terminations[element[2]] = True
                    self.truncated[element[2]] = False # To not have both terminated and trucated at the same time
                self.edge_data.pop(elm_i)
                self.busy[element[2]] = False
        self.state = self.next_state

    def _process_action(self, action, agent):
        if np.any(action > 0):  # tuples of row and columns with values
            indices = np.where(action > 0)[0]

            for j in indices:
                i = self.observations[agent][1]
                # Min value between people sent in action and actual people in the node.

                element = [(i, j), 0, agent]
                self.busy[agent] = True
                self.edge_data.append(element)

                self.next_state[i, i, 2] -= 1
                self.next_state[i, j, 2] += 1

    def _initialize_observations(self):
        agent_nr = 0

        for i, node in enumerate(np.diagonal(self.state[:, :, 2])):
            for j in range(int(node)):
                self.observations[self.agents[agent_nr]] = [self.state.copy(), i]
                agent_nr += 1

    def _update_observations(self):
        for agent in self.agents:
            self.observations[agent][0][:, :, 0] = self.state[:, :, 0] / self.max_dist
            self.observations[agent][0][:, :, 1] = self.state[:, :, 1] / self.max_band_widt
            self.observations[agent][0][:, :, 2] = self.state[:, :, 2] / self.n_agents

    def _update_masks(self, agent = None):
        if agent != None: #an agent is specified and not iterating over all of them
            self.action_masks[agent] = self.state[self.observations[agent][1], :, 0] > 0
            self.action_masks[agent][self.observations[agent][1]] = True
            if True in self.action_masks[agent]:
                self.action_masks[agent] = np.append(self.action_masks[agent], False) # no-op action
            else:
                self.action_masks[agent] = np.append(self.action_masks[agent], True) # no-op action
        else:
            for agent in self.agents:
                if self.busy[agent]:
                    self.action_masks[agent] = [False for _ in range(self.n_actions)]
                    self.action_masks[agent][-1] = True
                else:
                    self.action_masks[agent] = self.state[self.observations[agent][1], :, 0] > 0
                    self.action_masks[agent][self.observations[agent][1]] = True
                    self.action_masks[agent] = np.append(self.action_masks[agent], False) # no-op action





    def _initialize_matrix(self, map_name='very_small_map'):
        # GRAPH AND MATRIX INITIALIZATION
        search_place = False  # Modify when using a city or a file name
        load_data = False
        # map_name = 'Tronchetto'
        script_directory = os.path.dirname(__file__)
        data_dir = os.path.join(script_directory, 'data')

        osm_file_path = os.path.join(data_dir, f'osm/{map_name}.osm')
        matrix_file_path = os.path.join(data_dir, f'npy/{map_name}_matrix.npy')
        min_safe_distances_file_path = os.path.join(
            data_dir, f"txt/min_distances_to_safe_node_{map_name}.txt")
        safe_nodes_file_path = os.path.join(
            data_dir, f"txt/safe_nodes_{map_name}.txt")
        min_danger_distances_file_path = os.path.join(
            data_dir, f"txt/min_distances_to_danger_node_{map_name}.txt")
        danger_nodes_file_path = os.path.join(
            data_dir, f"txt/danger_nodes_{map_name}.txt")

        if not load_data and not os.path.exists(matrix_file_path) or not os.path.exists(min_safe_distances_file_path)\
                or not os.path.exists(safe_nodes_file_path):
            if search_place:
                G = ox.graph_from_place(
                    f'{map_name}, Venice, Italy', network_type='all')
            else:
                G = ox.graph_from_xml(osm_file_path)

            G = _clean_data(G)

            ox.plot_graph(G)

            # Safe and danger nodes
            nodes = list(range(len(G.nodes)))
            safe_nodes = [8] # random.sample(nodes, max(int(len(G.nodes) / 10), 1))
            remaining_nodes = list(set(nodes) - set(safe_nodes))
            self.danger_nodes = random.sample(
                remaining_nodes, max(int(len(remaining_nodes) / 10), 1))

            # Generate matrix
            self.matrix = generate_matrix(G, safe_nodes=safe_nodes,
                                     danger_nodes=self.danger_nodes)

            # Minimum distances
            self.min_distances_to_safe_node = _get_min_distances_to_safe_nodes(
                G, safe_nodes)
            self.min_distances_to_danger_node = _get_min_distances_to_danger_nodes(
                G, self.danger_nodes)

            # Store variables to memory
            with open(safe_nodes_file_path, 'w') as file:
                for item in safe_nodes:
                    file.write(f'{item}\n')
            with open(danger_nodes_file_path, 'w') as file:
                for item in self.danger_nodes:
                    file.write(f'{item}\n')

            np.save(matrix_file_path, self.matrix)

            with open(min_safe_distances_file_path, 'w') as file:
                for item in self.min_distances_to_safe_node:
                    file.write(f'{item}\n')

            with open(min_danger_distances_file_path, 'w') as file:
                for item in self.min_distances_to_danger_node:
                    file.write(f'{item}\n')

        else:
            # Load variables from memory
            self.safe_nodes = []
            with open(safe_nodes_file_path, 'r') as file:
                for line in file:
                    self.safe_nodes.append(int(line.strip()))
            self.danger_nodes = []
            with open(danger_nodes_file_path, 'r') as file:
                for line in file:
                    self.danger_nodes.append(int(line.strip()))

            self.matrix = np.load(matrix_file_path)
            self.matrix[:, :, 2] = 0


            self.min_distances_to_safe_node = []
            with open(min_safe_distances_file_path, 'r') as file:
                for line in file:
                    self.min_distances_to_safe_node.append(float(line.strip()))
            self.min_distances_to_danger_node = []
            with open(min_danger_distances_file_path, 'r') as file:
                for line in file:
                    self.min_distances_to_danger_node.append(float(line.strip()))


        self.G = ox.graph_from_xml(osm_file_path)

        self.G = _clean_data(self.G)