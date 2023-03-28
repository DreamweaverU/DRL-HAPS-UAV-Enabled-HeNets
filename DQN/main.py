"""
This Python code implements a Deep Q-Network (DQN) algorithm for aerial networks (composed of UAVs and HAPs) optimization.
DQN is a model-free reinforcement learning algorithm that uses a neural network to approximate the action-value
function. The neural network is trained using a dataset of experiences obtained by interacting with the environment.
The experiences are stored in a replay memory, and a random sample of experiences is used to update the neural network

The goal is to optimize the trajectory of the UAV and the channel allocation to maximize the network's performance
and meet the quality of service (QoS) requirements of the users.
Thus, UAVs learn optimal policies that maximize their total rewards which captures fairness and load
of aerial base stations (ABSs) over a finite time horizon.

@author: Atefeh Hajijamali Arani
@University of Waterloo
"""
#  ********************** import libraries **********************
from my_functions import *
from collections import defaultdict, namedtuple, deque
import random
from torch import nn
import torch
from collections import deque
import torch.optim as optim
import pandas as pd
import numpy as np

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):

    def __init__(self, in_features, out_features):
        super(DQN, self).__init__()

        self.model = nn.Sequential(nn.Linear(in_features, 256),
                                   nn.LayerNorm(256),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(256, 128),
                                   nn.LayerNorm(128),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(128, 64),
                                   nn.LayerNorm(64),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(64, 32),
                                   nn.LayerNorm(32),
                                   nn.ReLU(),
                                   nn.Dropout(0.2),
                                   nn.Linear(32, out_features),
                                   )

    def forward(self, x):
        return self.model(x.to(device)).to(device)

def select_action(state, policy_net,steps_done):
    sample = random.random()
    eps_threshold = epsilon_end + (epsilon_start - epsilon_end) * \
                    math.exp(-1. * steps_done / epsilon_decay)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            action = policy_net(state.type(torch.float32)).max(0)[1].view(1, 1)
            return action, steps_done # Here, shape is 28
            # return policy_net(state.type(torch.float32)).max(1)[1].view(1, 1) # If the shape is (1,28), we use max(dim=1)
    else:
        action = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        return action, steps_done

with open('config.yml', 'r') as file:
    config_yml = yaml.safe_load(file)

                #***********Define Gloabal Variables***********
hap_per_area = config_yml['Network']['number_of_HAP_per_area']
uav_per_area = config_yml['Network']['number_of_uavs_per_area']
total_ues = config_yml['Network']['total_users']
total_base_stations = hap_per_area + uav_per_area
print(f'Total users = {total_ues}')
print(f'Total UAVs = {uav_per_area}')
Earth_radius = config_yml['SAT_Network']['Earth_radius']
sat_height = config_yml['Base_Station']['height'][0]
hap_height = config_yml['Base_Station']['height'][1]

# Layout dimensions
area_width_x = config_yml['Network']['area_size'][0]
area_width_y = config_yml['Network']['area_size'][1]

# UAV Parameters
min_altitude_uav = config_yml['UAV_Network']['altitude_minimum']
max_altitude_uav = config_yml['UAV_Network']['altitude_maximum']
fixed_velocity_uav = config_yml['UAV_Network']['fixed_speed']
uav_height = config_yml['Base_Station']['height'][2]

# User Parameters
max_user_speed = config_yml['User']['maximum_speed']
ue_height = config_yml['User']['height']

# Ground BSs Link's Parameters for mmwave
alpha_1 = config_yml['Network']['alpha_1']
betta_1 = config_yml['Network']['betta_1']
alpha_2 = config_yml['Network']['alpha_2']
betta_2 = config_yml['Network']['betta_2']
std_los_2 = config_yml['Network']['std_los_2']

# UAVs Link's Parameters in mmwave
alpha_uav_los = config_yml['UAV_Network']['alpha_uav_los']
alpha_uav_nlos = config_yml['UAV_Network']['alpha_uav_nlos']
betta_uav_los = config_yml['UAV_Network']['betta_uav_los']
betta_uav_nlos = config_yml['UAV_Network']['betta_uav_nlos']
std_uav_los = config_yml['UAV_Network']['std_uav_los']
std_uav_nlos = config_yml['UAV_Network']['std_uav_nlos']

# Channel parameters
total_subcarriers_ground = config_yml['Network']['number_of_subcarriers_ground']
channels_set = list(range(0, total_subcarriers_ground, 1))
carrier_frequency = config_yml['Network']['carrier_frequency']
total_bandwidth = config_yml['Network']['bandwidth']
hap_carrier_freq = (config_yml['Network']['hap_carrier_freq']) * 1000
total_bandwidth_ground = config_yml['Network']['bandwidth_ground']
subcarrier_bandwidth_ground = total_bandwidth_ground / total_subcarriers_ground
total_subcarriers = config_yml['Network']['number_of_subcarriers']
subcarrier_bandwidth = total_bandwidth / total_subcarriers
noise = 10 ** ((config_yml['Network']['noise_power_ground'] - 30) / 10)
noise_backhaul = 10 ** ((config_yml['Network']['noise_power_backhaul'] - 30) / 10)

# UAV's action set parameters
action_count_uav = len([6, 0, 1, 2, 3, 4, 5]) * len(channels_set)  # Action length for UAVs
direction_actions_uav = [6, 0, 1, 2, 3, 4, 5]  # Direction movement for UAVs
x = np.array(np.meshgrid(direction_actions_uav, channels_set)).T.reshape(-1, 2)
y = (np.array(range(0, len(x), 1)))
y = y.reshape(len(y), 1)
total_actions_uav = np.concatenate([y, x], axis=1)  # The set of UAV's action
n_actions = total_actions_uav.shape[0]

# Transmit power parameters
power_sat_max = config_yml['Base_Station']['power_maximum'][0] - 30 # Maximum power of LEO
power_sat_max_wat = to_watts(power_sat_max)
power_hap_max = config_yml['Base_Station']['power_maximum'][1] - 30 # Maximum power of HAP
power_uav_max = config_yml['Base_Station']['power_maximum'][2] - 30 # Maximum power of UAV
power_hap_max_all_hap = to_watts(power_hap_max) * np.ones((hap_per_area, 1))
power_uav_max_all_uav = to_watts(power_uav_max) * np.ones((uav_per_area, 1))
power_max_combined = np.append(power_hap_max_all_hap, power_uav_max_all_uav).reshape(1, hap_per_area + uav_per_area)

# Iteration related parameters
total_time = config_yml['Algorithm']['iterations_for_time']
total_iteration = config_yml['Algorithm']['total_iteration_monte']
time_resolution = config_yml['Algorithm']['time_resolution']

#    ******************** Initialize_Buffer_Learning *************************
channel_uav = np.zeros((uav_per_area, total_time))
channel_hap = np.ones((hap_per_area, total_time)) * len(channels_set)

user_throughput_individual_learning_outage_error = np.zeros((total_ues, total_time))
user_throughput_individual_learning = np.zeros((total_ues, total_time))
user_throughput_mean_learning_outage_error = np.zeros((total_time, 1))
user_throughput_mean_learning = np.zeros((1, total_time))
user_load_individual_learning = np.zeros((total_ues, total_time))
served_user_indicator_learning = np.zeros((total_ues, total_time))  # It's an indicator if a UE is served by a BS it will change to 1
user_arrival_rate_requierment_learning = config_yml['User']['mean_packet_size'] * 10 ** 6 * np.ones((total_ues, total_time))
active_users_learning = -np.ones((1, total_time))
active_mean_users_learning = -np.ones((1, total_time))

base_station_throughput_learning_outage_error = np.zeros((total_base_stations, total_time))
base_station_throughput_learning = np.zeros((total_base_stations, total_time))
base_station_load_learning_outage_error = np.zeros((total_base_stations, total_time))
base_station_load_learning = np.zeros((total_base_stations, total_time))
base_station_estimated_load_learning = np.zeros((total_base_stations, total_time))
user_throughput_individual_learning_time_average = np.zeros((total_ues, total_time))
user_throughput_mean_learning_time_average = np.zeros((1, total_time))
base_station_load_learning_time_average = np.zeros((total_base_stations, total_time))
bs_load_preferred_combined = 0.5 * np.ones((total_base_stations, 1))

base_station_utility_learning_outage_error = np.zeros((total_base_stations, total_time))
base_station_utility_learning = np.zeros((total_base_stations, total_time))
base_station_utility_learning_time_average_outage_error = np.zeros((total_base_stations, total_time))
base_station_utility_learning_time_average = np.zeros((total_base_stations, total_time))

serving_cell_learning = -np.ones((total_ues, total_time))

mean_load_time = np.zeros(total_iteration)
mean_rate_time_learning = np.zeros(total_iteration)
mean_drop_time = np.zeros(total_iteration)
mean_utility_time = torch.zeros(total_iteration)
mean_throughput = np.zeros(total_iteration)
mean_fairness_time = np.zeros(total_iteration)

# DQN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
gamma = config_yml['dqn_param']['gamma']
batch_size = config_yml['dqn_param']['batch_size']
replay_memory_size = config_yml['dqn_param']['replay_memory_size']
min_replay_size = config_yml['dqn_param']['min_replay_size']
epsilon_start = config_yml['dqn_param']['epsilon_start']
epsilon_end = config_yml['dqn_param']['epsilon_end']
epsilon_decay = config_yml['dqn_param']['epsilon_decay']
target_update_freq = config_yml['dqn_param']['target_update_freq']
replay_buffer = deque(maxlen=replay_memory_size)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
DQN_uavs = namedtuple('DQN_uavs', ('policy_net', 'target_net', 'optimizer', 'memory'))
list_DQN_uavs = []

for _ in range(uav_per_area):
    policy_net = DQN(7, n_actions).to(device)
    target_net = DQN(7, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)
    memory = ReplayMemory(replay_memory_size)
    bb = DQN_uavs(policy_net, target_net, optimizer, memory)
    list_DQN_uavs.append(bb)

steps_done_uavs = torch.zeros(uav_per_area)

# ********************* Main Algorithm ***************************

for rr in range(total_iteration): # Montecarlo iteration
    cart_haps, cart_uavs, cart_ues = make_layout_uniform()   # Create the geometric layout of an aerial heterogeneous network
    drop_ues_vector_count_learning = np.zeros((total_base_stations, total_time))  # Number of outage users for each BS
    server_ues_vector_count_learning = np.zeros((total_base_stations, total_time))  # Number of users served by each BS
    utility_bss = torch.zeros((total_base_stations, total_time))  # Utility of BSs

    for time_step in range(total_time):
        cart_ues = mobility_user(cart_ues, time_resolution)  # Update users' locations
        previous_cart_uavs = cart_uavs.copy()

    # Initialization for each montecarlo simulation
        if time_step == 0:
            fairness = np.zeros(total_time)
            previous_channel_uav = -np.ones((uav_per_area, total_time))
            channel_uav[:, time_step] = np.random.randint(0, len(channels_set), (uav_per_area, 1)).squeeze()
            previous_action_uav = np.random.randint(0, total_actions_uav.shape[0], (uav_per_area, 1)).squeeze()
            previous_channel_uav[:, time_step] = total_actions_uav[previous_action_uav, 2].copy()
            action_current_uav = - torch.ones((uav_per_area, total_time))
            direction_movement_uavs = -np.ones((uav_per_area, total_time))

            previous_serving_cell_learning = -np.ones((total_ues, 1))
            previous_served_user_indicator_learning = np.zeros((total_ues, 1))
            serving_cell_learning = -np.ones((total_ues, total_time))
            serving_user_learning = defaultdict(list)
            serving_user_learning = {key: [[] for _ in range(3)] for key in np.arange(0, total_base_stations, 1)}
            current_channel_uav = total_actions_uav[previous_action_uav, 2].copy()
            cart_uavs_normalized, channel_uav_one_hot = state_standardization(previous_cart_uavs, current_channel_uav)
            state_uavs = torch.cat((torch.tensor(cart_uavs_normalized), channel_uav_one_hot.reshape(uav_per_area, 4)), dim=1)
        else:
            previous_action_uav = action_current_uav[:, time_step - 1].clone()
            previous_channel_uav[:, time_step] = channel_uav[:, time_step - 1].copy()
            previous_serving_cell_learning = serving_cell_learning[:, time_step - 1].copy()
            previous_served_user_indicator_learning = served_user_indicator_learning[:, time_step - 1].copy()

        previous_associated_user_counter_learning = [len(serving_user_learning[bs][0]) for bs in
                                                     range(total_base_stations)]
    # Action selection
        for uav in range(uav_per_area):
            action_current_uav[uav, time_step], steps_done_uavs[uav] = select_action(state_uavs[uav, :], list_DQN_uavs[uav].policy_net, steps_done_uavs[uav])

    # Find movement direction of UAVs based on their actions
        for uav in range(uav_per_area):
            action_index = action_current_uav[uav, time_step].clone()
            direction_movement_uavs[uav, time_step] = total_actions_uav[int(action_index), 1]
            cart_uavs[uav, :] = location_update(direction_movement_uavs[uav, time_step], fixed_velocity_uav,
                                                cart_uavs[uav, :], time_resolution)

        distance_BSs_ues_2D, distance_BSs_ues_3D, path_loss_bs_ue_all_with_fading_learning = calculate_pathloss(
            cart_ues, cart_haps, cart_uavs)

        channel_bs_learning = np.concatenate([channel_hap[:, time_step], channel_uav[:, time_step]], axis=0)
        if time_step == 0:
            previous_outage_indicator_learning = np.zeros(total_base_stations)
            outage_base_station_indicator_learning = np.zeros((total_base_stations, total_time))
        else:
            previous_outage_indicator_learning = outage_base_station_indicator_learning[:, time_step - 1]

        # User-ABS association
        serving_cell_learning[:, time_step], serving_user_learning = find_serving_cells(
            path_loss_bs_ue_all_with_fading_learning,
            power_max_combined,
            previous_serving_cell_learning,
            previous_served_user_indicator_learning,
            serving_user_learning
        )
        serving_user_learning = schedule_users(user_arrival_rate_requierment_learning[:, time_step], serving_user_learning)

        serving_user_learning = fixed_point_iterations(channel_bs_learning, \
                                                       path_loss_bs_ue_all_with_fading_learning, \
                                                       power_max_combined, \
                                                       serving_user_learning, bs_load_preferred_combined)

        user_throughput_individual_learning_outage_error[:, time_step], user_load_individual_learning[:, time_step], \
        user_throughput_mean_learning_outage_error[time_step], base_station_throughput_learning_outage_error[:,
                                                               time_step], base_station_load_learning_outage_error[:,
                                                                           time_step] = calculate_rate_load(serving_user_learning)

        fairness[time_step], drop_ues_vector_count_learning[:, time_step], base_station_load_learning[:, time_step], \
        base_station_throughput_learning[:, time_step], user_load_individual_learning[:, time_step], \
        user_throughput_individual_learning[:, time_step], served_user_indicator_learning[:, time_step] = \
            compensate_outage(serving_user_learning, previous_associated_user_counter_learning, \
                              user_arrival_rate_requierment_learning[:, time_step], \
                              base_station_load_learning_outage_error[:, time_step],
                              base_station_throughput_learning_outage_error[:, time_step],
                              user_load_individual_learning[:, time_step],
                              user_throughput_individual_learning_outage_error[:, time_step]
                              )

        outage_base_station_indicator_learning[:, time_step] = find_outage_baseStations(
            served_user_indicator_learning[:, time_step], serving_cell_learning[:, time_step], previous_serving_cell_learning)

    # Calculate utility
        utility_bss_current = utility_base_station(fairness[time_step], base_station_load_learning[:, time_step])
        utility_uavs = utility_bss_current[1:].unsqueeze(-1)
        utility_hap = utility_bss_current[0].unsqueeze(-1)
        utility_bss[:, time_step] = utility_bss_current.clone()
        cart_uavs_normalized_next, channel_uav_one_hot_next = state_standardization(cart_uavs, \
                                                                                    total_actions_uav[action_current_uav[:, time_step].type(torch. int64), 2])
        next_state_uav = torch.cat((torch.tensor(cart_uavs_normalized_next), \
                                    channel_uav_one_hot_next.reshape(uav_per_area, 4)), dim=1)
        for uav in range(uav_per_area):
            list_DQN_uavs[uav].memory.push(state_uavs[uav, :].reshape(1,7), action_current_uav[uav, time_step].unsqueeze(-1), next_state_uav[uav,:].reshape(1,7), utility_uavs[uav])
            state_uavs[uav,:] = next_state_uav[uav,:]
            optimize_model(list_DQN_uavs[uav].memory, Transition, list_DQN_uavs[uav].policy_net.to(device), list_DQN_uavs[uav].target_net.to(device), list_DQN_uavs[uav].optimizer)
            if time_step % target_update_freq == 0:
                list_DQN_uavs[uav].target_net.load_state_dict(list_DQN_uavs[uav].policy_net.state_dict())

        base_station_load_learning_time_average = calcuate_average(base_station_load_learning[:, time_step], \
                                                                   base_station_load_learning_time_average, time_step)
        user_throughput_individual_learning_time_average = calcuate_average(
            user_throughput_individual_learning[:, time_step], \
            user_throughput_individual_learning_time_average, time_step)

    mean_load_time[rr] = np.mean(base_station_load_learning_time_average[:, -1])
    mean_rate_time_learning[rr] = np.mean(user_throughput_individual_learning_time_average[:, -1])
    mean_drop_time[rr] = np.mean(drop_ues_vector_count_learning)
    mean_utility_time[rr] = torch.mean(utility_bss)
    mean_throughput[rr] = np.mean(base_station_throughput_learning)
    mean_fairness_time[rr] = np.mean(fairness)

    if rr % 25 == 0:
        print(f'rr = {rr}')
# ******************* Simulation Results *******************************

avg_load = np.mean(mean_load_time)
avg_rate = np.mean(mean_rate_time_learning)
avg_drop = np.mean(mean_drop_time)
avg_utility = torch.mean(mean_utility_time)
avg_throughput = np.mean(mean_throughput)
avg_fairness = np.mean(mean_fairness_time)
print(f'avg_load = {avg_load}')
print(f'avg_rate = {avg_rate}')
print(f'avg_drop = {avg_drop}')
print(f'avg_utility = {avg_utility}')
print(f'avg_throughput = {avg_throughput}')
print(f'avg_fairness = {avg_fairness}')

saved_results = {'avg_load' : [avg_load],  'avg_rate': [avg_rate], 'avg_drop': [avg_drop], \
                 'avg_utility' : [avg_utility.item()], 'avg_throughput' : [avg_throughput], 'avg_fairness' : [avg_fairness]}

results = pd.DataFrame.from_dict(saved_results)
results.to_csv('results.csv' , sep='\t', index = False)


