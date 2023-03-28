"""
Q-Learning method for UAV trajectory design and channel allocation optimization
@author: Atefeh Hajijamali Arani
@Project: HAPS-UAV-Enabled Heterogeneous Networks:
          A Deep Reinforcement Learning Approach
@University of Waterloo
"""
#  ********************** import libraries **********************
import yaml
#import pdb
import numpy as np
from my_functions import *
from collections import defaultdict, namedtuple, deque
import random
from torch import nn
import torch
from collections import deque
import itertools
from torch import functional as f
import torch.optim as optim
import csv
import pandas as pd



with open('config.yml', 'r') as file:
    config_yml = yaml.safe_load(file)
# pdb.set_trace()
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
action_count_uav = len([6, 2, 3, 4, 5])
direction_actions_uav = [6, 2, 3, 4, 5]
x = np.array(direction_actions_uav)
y = (np.array(range(0, len(x), 1)))
y = y.reshape(len(y), 1)
total_actions_uav = np.concatenate([y, x.reshape(-1, 1)], axis=1)


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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epsilon = config_yml['qlearning_algorithm']['explore_probability']
lr_q = config_yml['qlearning_algorithm']['learning_rate']
gamma_q = config_yml['qlearning_algorithm']['gamma']
n_actions = total_actions_uav.shape[0]
total_horizontal_locations_uav = Generate_set_of_locations(fixed_velocity_uav)
len_total_locations = len(total_horizontal_locations_uav)
q_tables_uav = []
q_next_max_uav = np.zeros(uav_per_area)
q_current_uav = np.zeros(uav_per_area)

for _ in range(uav_per_area):
    q_tables_uav.append(np.zeros((len_total_locations, n_actions)))


def select_action(state, q_tables_uav_individual):
    sample = random.random()
    if sample > epsilon:
        # Exploitation
        action = np.argmax(q_tables_uav_individual[int(state)])
        return action # Here, shape is 28
    else:
        # Exploration
        action = random.randrange(len(total_actions_uav))
        return action

# ********************* Main Algorithm ***************************

for rr in range(total_iteration): # Montecarlo iteration
    state = np.zeros((uav_per_area, total_time))
    next_state_uav = np.zeros(uav_per_area)
    cart_haps, cart_uavs, cart_ues = make_layout_uniform()
    cart_uavs[:,2] = max_altitude_uav
    drop_ues_vector_count_learning = np.zeros((total_base_stations, total_time))
    server_ues_vector_count_learning = np.zeros((total_base_stations, total_time))
    utility_bss = torch.zeros((total_base_stations, total_time))
    for time_step in range(total_time):
        cart_ues = mobility_user(cart_ues, time_resolution)
        previous_cart_uavs = cart_uavs.copy()

        if time_step == 0:
            fairness = np.zeros(total_time)
            previous_channel_uav = -np.ones((uav_per_area, total_time))

            channel_uav[:, time_step] = np.random.randint(0, len(channels_set), (uav_per_area, 1)).squeeze()
            previous_action_uav = np.random.randint(0, total_actions_uav.shape[0],
                                                    (uav_per_area, 1)).squeeze()
            previous_channel_uav[:, time_step] = random.choice(channels_set)
            action_current_uav = - torch.ones((uav_per_area, total_time))
            direction_movement_uavs = -np.ones((uav_per_area, total_time))

            previous_serving_cell_learning = -np.ones((total_ues, 1))
            previous_served_user_indicator_learning = np.zeros((total_ues, 1))
            serving_cell_learning = -np.ones((total_ues, total_time))
            serving_user_learning = defaultdict(list)
            serving_user_learning = {key: [[] for _ in range(3)] for key in np.arange(0, total_base_stations, 1)}

        else:

            previous_action_uav = action_current_uav[:, time_step - 1].clone()
            previous_channel_uav[:, time_step] = channel_uav[:, time_step - 1].copy()
            previous_serving_cell_learning = serving_cell_learning[:, time_step - 1].copy()
            previous_served_user_indicator_learning = served_user_indicator_learning[:, time_step - 1].copy()

        previous_associated_user_counter_learning = [len(serving_user_learning[bs][0]) for bs in
                                                     range(total_base_stations)]


        for uav in range(uav_per_area):
            state[uav, time_step] = np.where((total_horizontal_locations_uav == cart_uavs[uav, 0:2]).all(axis=1))[0]
            action_current_uav[uav, time_step] = select_action(state[uav, time_step], q_tables_uav[uav])

            action_index = action_current_uav[uav, time_step].clone()
            direction_movement_uavs[uav, time_step] = total_actions_uav[int(action_index), 1]
            cart_uavs[uav, :] = location_update(direction_movement_uavs[uav, time_step], fixed_velocity_uav,
                                                cart_uavs[uav, :], time_resolution)

        distance_BSs_ues_2D, distance_BSs_ues_3D, path_loss_bs_ue_all_with_fading_learning = calculate_pathloss(
            cart_ues, cart_haps, cart_uavs)

        channel_bs_learning = np.concatenate([channel_hap[:, time_step], channel_uav[:, time_step]],
                                             axis=0)
        if time_step == 0:
            previous_outage_indicator_learning = np.zeros(total_base_stations)
            outage_base_station_indicator_learning = np.zeros((total_base_stations, total_time))
        else:
            previous_outage_indicator_learning = outage_base_station_indicator_learning[:, time_step - 1]

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
        for uav in range(uav_per_area):
            # Update Q-tables
            next_state_uav[uav] = np.where((total_horizontal_locations_uav == cart_uavs[uav, 0:2]).all(axis=1))[0]
            q_current_uav[uav] = q_tables_uav[uav][int(state[uav, time_step])][int(action_current_uav[uav, time_step])]
            q_next_max_uav[uav] = np.max(q_tables_uav[uav][int(next_state_uav[uav])])
            q_tables_uav[uav][int(state[uav, time_step])][int(action_index)] = q_current_uav[uav] + lr_q * (\
                    utility_uavs[uav] + gamma_q * q_next_max_uav[uav] - q_current_uav[uav])


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
#time_elapsed = time.time() - since

saved_results = {'avg_load' : [avg_load],  'avg_rate': [avg_rate], 'avg_drop': [avg_drop],\
                 'avg_utility' : [avg_utility.item()], 'avg_throughput' : [avg_throughput], 'avg_fairness' : [avg_fairness]}

results = pd.DataFrame.from_dict(saved_results)
results.to_csv('results.csv' , sep='\t', index = False)


