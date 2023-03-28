import numpy as np
import yaml
import math
from scipy.spatial import distance
from numpy import random
import random

from torch import nn
import torch
from collections import deque
import itertools
from torch import functional as f

with open('config.yml', 'r') as file:
    config_yml = yaml.safe_load(file)
 #***********Define Gloabal Variables***********
hap_per_area = config_yml['Network']['number_of_HAP_per_area']
uav_per_area = config_yml['Network']['number_of_uavs_per_area']
total_ues = config_yml['Network']['total_users']
total_base_stations = hap_per_area + uav_per_area
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

# DQN parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gamma = config_yml['dqn_param']['gamma']
batch_size = config_yml['dqn_param']['batch_size']
replay_memory_size = config_yml['dqn_param']['replay_memory_size']
min_replay_size = config_yml['dqn_param']['min_replay_size']
epsilon_start = config_yml['dqn_param']['epsilon_start']
epsilon_end = config_yml['dqn_param']['epsilon_end']
epsilon_decay = config_yml['dqn_param']['epsilon_decay']
target_update_freq = config_yml['dqn_param']['target_update_freq']
replay_buffer = deque(maxlen=replay_memory_size)

# Convert dB to Watts
def to_watts(in_dBs):
    in_watts = 10 ** (in_dBs / 10)
    return in_watts

# Generate the set of locations in the layout
# Input: UAV's velocity,       Output: a set of locations that are inside the layout
def Generate_set_of_locations(fixed_velocity_uav):
    x = np.arange(- area_width_x/2, area_width_x/2 + fixed_velocity_uav, fixed_velocity_uav)
    y = np.arange(- area_width_y/2, area_width_y/2 + fixed_velocity_uav, fixed_velocity_uav)
    all_points = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
    index_inside_points = [check_inside(point) for point in all_points]
    inside_points = all_points[index_inside_points]
    return inside_points

# Initialize locations of users and BSs
# The function uses a uniform distribution to randomly generate the locations of the users.
def make_layout_uniform():
    np.random.seed(42)
    cart_ue = np.zeros((total_ues, 2))
    for ue in range(total_ues):
        x_temp_ue = - area_width_x / 2 + (area_width_x * np.random.uniform(0, 1, 1))
        y_temp_ue = - area_width_y / 2 + (area_width_y * np.random.uniform(0, 1, 1))
        cart_ue[ue, 0] = x_temp_ue
        cart_ue[ue, 1] = y_temp_ue
    if hap_per_area > 0:
        cart_hap = np.zeros((hap_per_area, 3))
        cart_hap[:, 0] = 0
        cart_hap[:, 1] = 0
        cart_hap[:, 2] = hap_height * np.ones((hap_per_area, 1))
    else:
        cart_hap = 'None'

    cart_uav = np.zeros((uav_per_area, 3))
    total_horizontal_locations_uav = Generate_set_of_locations(fixed_velocity_uav)
    temp_total_horizontal_locations_uav = total_horizontal_locations_uav.copy()
    temp_total_points_inside = len(total_horizontal_locations_uav)

    cart_bss_uavs = cart_hap[0,0:2].copy().reshape(1,2)

    for uav in range(uav_per_area):
        distance_point_bs = np.ones((temp_total_points_inside , uav+1))
        for point in range(len(temp_total_horizontal_locations_uav)):
            distance_point_bs[point, :] = [distance.euclidean(temp_total_horizontal_locations_uav[point, :], cart_bss_uavs[i, :]) for i in range(cart_bss_uavs.shape[0])]
        min_distance_point_bs = np.min(distance_point_bs, 1)
        max_min_index = np.argmax(min_distance_point_bs)
        temp_horizontal_uav = temp_total_horizontal_locations_uav[max_min_index, :]
        temp_total_horizontal_locations_uav = np.delete(temp_total_horizontal_locations_uav, max_min_index, 0)
        cart_bss_uavs = np.vstack((cart_bss_uavs, temp_horizontal_uav))
        cart_uav[uav, 0:2] = temp_horizontal_uav
        temp_total_points_inside = len(temp_total_horizontal_locations_uav)
    cart_uav[:, 2] = min_altitude_uav
    return cart_hap, cart_uav, cart_ue

# Update locations of users based on a random mobility model
def mobility_user(cart_ue, time_resolution):
    np.random.seed(42)
    new_cart_temp = np.zeros((total_ues, 2))
    for ue in range(total_ues):
        flag = 0
        while flag == 0:
            speed_ue = max_user_speed * time_resolution * np.random.rand(1)
            angle_ue = 2 * np.pi * np.random.rand(1)
            new_cart_temp[ue, 0] = cart_ue[ue, 0] + speed_ue * np.cos(angle_ue)
            new_cart_temp[ue, 1] = cart_ue[ue, 1] + speed_ue * np.sin(angle_ue)
            check_est = check_inside(new_cart_temp[ue, :])
            if check_est:
                flag = 1

    return new_cart_temp

# Check if a given coordinate is within the bounds of the layout
def check_inside(point):
    abs_point = np.abs(point)
    check_1 = abs_point[0] <= area_width_x / 2
    check_2 = abs_point[1] <= area_width_y / 2
    check = (check_1 & check_2)
    return check

# Update the locations of UAVs based on direction movements, speed
def location_update(direction, velocity, cart_uav, delta_t):
    cart_uav_temp = cart_uav.copy()
    if direction == 6:
        cart_uav_temp = cart_uav

    elif direction == 0:  # Up
        cart_uav_temp[2] = cart_uav[2] + velocity * delta_t
        if cart_uav_temp[2] > max_altitude_uav:
            cart_uav_temp[2] = max_altitude_uav

    elif direction == 1:  # Down
        cart_uav_temp[2] = cart_uav[2] - velocity * delta_t
        if cart_uav_temp[2] < min_altitude_uav:
            cart_uav_temp[2] = min_altitude_uav

    elif direction == 2:
        cart_uav_temp[0] = cart_uav[0] + velocity * delta_t
        point_inside_indicator = check_inside(cart_uav_temp)
        if not point_inside_indicator:
            cart_uav_temp[0] = cart_uav[0]

    elif direction == 3:
        cart_uav_temp[0] = cart_uav[0] - velocity * delta_t
        point_inside_indicator = check_inside(cart_uav_temp)
        if not point_inside_indicator:
            cart_uav_temp[0] = cart_uav[0]

    elif direction == 4:
        cart_uav_temp[1] = cart_uav[1] + velocity * delta_t
        point_inside_indicator = check_inside(cart_uav_temp)
        if not point_inside_indicator:
            cart_uav_temp[1] = cart_uav[1]

    elif direction == 5:
        cart_uav_temp[1] = cart_uav[1] - velocity * delta_t
        point_inside_indicator = check_inside(cart_uav_temp)
        if not point_inside_indicator:
            cart_uav_temp[1] = cart_uav[1]
    return cart_uav_temp

# Calculate the path loss between ABSs and users by taking into account the probability of LoS
def calculate_pathloss(cart_ues, cart_haps, cart_uavs):
    a_1 = 0.1
    a_2 = 750
    a_3 = 8
    distance_bs_ue_2D = np.zeros((total_base_stations, total_ues))
    distance_bs_ue_3D = np.zeros((total_base_stations, total_ues))
    r_parameter_matrix = np.zeros((total_base_stations, total_ues))
    probability_of_LoS = np.ones((total_base_stations, total_ues))
    path_loss_bs_ue_all = np.zeros((total_base_stations, total_ues))

    if hap_per_area > 0:
        cart_all_bs = np.concatenate((cart_haps[:, :], cart_uavs[:, :]), axis=0)
        all_bss_hight = np.concatenate((cart_haps[:, 2], cart_uavs[:, 2]), axis=0)
    else:
        cart_all_bs = cart_uavs[:, :]

    for ue in range(total_ues):
        distance_bs_ue_2D[:, ue] = [distance.euclidean(cart_ues[ue, :], cart_all_bs[bs, :2]) for bs in
                                    range(total_base_stations)]
        distance_bs_ue_3D[:, ue] = [distance.euclidean(np.append(cart_ues[ue, :], ue_height), cart_all_bs[bs, :]) for bs
                                    in range(total_base_stations)]

        path_loss_bs_ue_all[:hap_per_area, ue] = [
            32.44 + 20 * np.log10(distance_bs_ue_3D[bs, ue] / 1000) + 20 * np.log10(hap_carrier_freq)
            for bs in range(hap_per_area)]

        r_parameter_matrix[:, ue] = np.floor(((distance_bs_ue_2D[:, ue] * np.sqrt(a_1 * a_2)) / 1000) - 1)
        for bs in range(1, total_base_stations):

            for j in range(0, int(r_parameter_matrix[bs, ue]) + 1):
                exp_input = -(all_bss_hight[bs] - (
                        (j + 0.5) * (all_bss_hight[bs] - ue_height) / (r_parameter_matrix[bs, ue] + 1))) ** 2 / (
                                    2 * (a_3 ** 2))
                probability_of_LoS[bs, ue] = probability_of_LoS[bs, ue] * (1 - np.exp(exp_input))
                probability_of_LoS[0, ue] = 1
            np.random.seed(55)
            random_number = np.random.rand()

            if random_number <= probability_of_LoS[bs, ue]:
                path_loss_bs_ue_all[bs, ue] = alpha_uav_los + betta_uav_los * 10 * np.log10(
                    distance_bs_ue_3D[bs, ue]) - np.random.randn() * std_uav_los
            else:
                path_loss_bs_ue_all[bs, ue] = alpha_uav_los + 3 * 10 * np.log10(
                    distance_bs_ue_3D[bs, ue]) - np.random.randn() * std_uav_nlos

    return distance_bs_ue_2D, distance_bs_ue_3D, path_loss_bs_ue_all

# ABSs and user association based on the maximum RSS
def find_serving_cells(path_loss_all,
                       maximum_transmission_power,
                       previous_associated_bs,
                       previous_served_indicator,
                       previous_serving_user_combined,
                       ):
    serving_cell_combined = previous_associated_bs.copy()
    serving_user_combined = previous_serving_user_combined.copy()

    transmit_power_with_indicator = 10 * np.log10(maximum_transmission_power)
    active_users = np.arange(0, total_ues, 1)
    previously_unserved_active_users = np.where(np.array(previous_served_indicator) == 0)[0]

    for user in range(len(previously_unserved_active_users)):
        ue = previously_unserved_active_users[user]
        path_loss_ue = path_loss_all[:, ue]
        association_metric = transmit_power_with_indicator.squeeze() - path_loss_ue

        if previous_associated_bs[ue] > -1:
            index_bs = previous_associated_bs[ue]
            association_metric[int(index_bs)] = -np.inf

        if np.isinf(association_metric).all():
            serving_cell_combined[ue] = previous_associated_bs[ue]
        else:
            serving_cell_combined[ue] = np.argmax(association_metric)

        if previous_associated_bs[ue] > -1:
            serving_user_combined[previous_associated_bs[ue]][0].remove(ue)
        serving_user_combined[int(serving_cell_combined[ue])][0].append(ue)
    return serving_cell_combined.squeeze(), serving_user_combined

# User scheduling based on equal share of resource
def schedule_users(ue_packet_arrivals, serving_users):
    for bs in range(total_base_stations):
        if len(serving_users[bs][0]) > 0:
            serving_users[bs][1] = (ue_packet_arrivals[serving_users[bs][0]])
            serving_users[bs][2] = (1 / len(serving_users[bs][0]) * np.ones((1, len(serving_users[bs][0]))))
    return serving_users

#  Fixed-point iterations for load estimation and resource allocation
def fixed_point_iterations(channel_vector, path_loss_all, maximum_transmission_power, serving_user_arrival,
                           base_station_load_estimation):
    transmission_power = maximum_transmission_power
    for fixed_point_iteration in range(5):
        base_station_load_estimation_old = np.minimum(base_station_load_estimation, 1)
        load_estimation_with_outage = base_station_load_estimation.copy()
        for bs in range(total_base_stations):
            associated_ues = serving_user_arrival[bs][0].copy()
            if len(associated_ues) == 0:
                base_station_load_estimation[bs] = 0
                continue
            path_gain_to_ues = np.zeros((1, len(associated_ues)))
            path_loss_to_ue = np.zeros((total_base_stations, len(associated_ues)))
            for ue in range(len(associated_ues)):
                path_gain_to_ues[0, ue] = to_watts(-path_loss_all[bs, associated_ues[ue]])
                path_loss_to_ue[:, ue] = to_watts(-path_loss_all[:, associated_ues[ue]])
            path_loss_to_ue_temp = path_loss_to_ue.copy()
            base_station_load_estimation_old_temp = base_station_load_estimation_old.copy()

            transmission_power_temp = transmission_power.copy()
            base_station_load_estimation_old_temp = base_station_load_estimation_old.copy()
            a = np.where(channel_vector == channel_vector[bs])[0]
            a = np.delete(a, np.where(a == bs))

            path_loss_to_ue_temp = path_loss_to_ue_temp[a, :]
            transmission_power_temp = transmission_power_temp[0, a]
            base_station_load_estimation_old_temp = base_station_load_estimation_old_temp[a]

            if len(path_loss_to_ue_temp) == 0 and len(base_station_load_estimation_old_temp) == 0:
                interference_to_ues_temp = np.zeros((1, path_loss_to_ue_temp.shape[1]))
            else:
                power_load_product = transmission_power_temp * base_station_load_estimation_old_temp.squeeze()
                interference_to_ues_temp = np.matmul(path_loss_to_ue_temp.T, power_load_product.reshape(len(power_load_product),1))
            rate_per_user = subcarrier_bandwidth_ground * np.log2(
                1 + transmission_power.squeeze()[bs] * path_gain_to_ues.squeeze() /
                (noise * subcarrier_bandwidth_ground + interference_to_ues_temp.squeeze()))

            arrivals_per_user = serving_user_arrival[bs][1]
            load_per_user = np.array(arrivals_per_user) / rate_per_user

            base_station_load_estimation[bs] = np.sum(load_per_user.squeeze())
            serving_user_arrival[bs][2] = load_per_user.copy()

        compare_1 = load_estimation_with_outage.copy()
        compare_1[np.isinf(load_estimation_with_outage)] = 1
        compare_2 = base_station_load_estimation.copy()
        compare_2[np.isinf(base_station_load_estimation)] = 1
        if (np.linalg.norm(compare_1 - compare_2) <= 1e-24):
            break
    return serving_user_arrival

# Estimate load of ABSs
def load_estimation_rule_for_basestations(time_step, all_time_load_estimations, all_time_handled_load):
    learning_rate = (1 / (time_step+ 1)) ** (0.9)
    new_all_time_load_estimations = all_time_load_estimations.copy()
    bs_load_preferred_combined = 0.5 * np.ones((total_base_stations, 1))
    if time_step == 0:
        new_estimation = bs_load_preferred_combined.copy()
    else:
        old_estimation = all_time_load_estimations[:, time_step - 1].copy()
        previous_load = np.clip(all_time_handled_load[:, time_step - 1], a_min=0, a_max=1)
        new_estimation = (1 - learning_rate) * old_estimation + (learning_rate) * previous_load
    new_all_time_load_estimations[:, time_step] = new_estimation.squeeze().copy()
    return new_all_time_load_estimations

# Calculate load and rate for users associated to ABSs
def calculate_rate_load(serving_user_rateNeed_resource_partition):
    individual_user_throughput = np.zeros(total_ues)
    individual_user_load = np.zeros(total_ues)
    cell_throughput = np.zeros(total_base_stations)
    cell_load = np.zeros(total_base_stations)
    for bs in range(total_base_stations):
        associated_ues = serving_user_rateNeed_resource_partition[bs][0]
        if len(associated_ues) > 0:
            cell_load[bs] = np.sum(serving_user_rateNeed_resource_partition[bs][2])
            individual_user_load[associated_ues] = serving_user_rateNeed_resource_partition[bs][2]
            individual_user_throughput[associated_ues] = serving_user_rateNeed_resource_partition[bs][1] / cell_load[bs]
            cell_throughput[bs] = np.sum(individual_user_throughput[associated_ues])
    mean_user_throughput = np.mean(individual_user_throughput)
    cell_load[np.isinf(cell_load)] = 1 + .1
    return individual_user_throughput.squeeze(), individual_user_load.squeeze(), mean_user_throughput.squeeze(), cell_throughput.squeeze(), cell_load.squeeze()

# Calculate the performance metrics for users and ABSs that are experiencing outages
def compensate_outage(serving_user_rate_need_resource_partition, original_associated_user_count, ue_packet_arrivals,
                      bs_load_combined, bs_rate_combined, ue_load_combined, ue_rate_combined):
    served_ue_indicator = np.zeros(total_ues)
    actual_bs_load = bs_load_combined.copy()
    actual_bs_rate = bs_rate_combined.copy()
    actual_ue_load = ue_load_combined.copy()
    actual_ue_rate = ue_rate_combined.copy()
    server_ues_count = np.zeros(total_base_stations)
    dropped_ues_count = np.zeros(total_base_stations)
    associated_ues_count = np.zeros(total_base_stations)
    for bs in range(total_base_stations):
        associated_ues = serving_user_rate_need_resource_partition[bs][0].copy()
        sorted_served_ues_index = associated_ues
        if len(associated_ues) > 0:
            associated_ues_load = ue_load_combined[associated_ues].copy()
            if bs_load_combined[bs] > 1:
                new_associated_user_count = len(associated_ues)
                if new_associated_user_count > original_associated_user_count[bs]:
                    sorted_original_associated_ues_load = np.sort(
                        associated_ues_load[0:original_associated_user_count[bs]], axis=None)
                    sorted_original_associated_ues_index = np.argsort(
                        associated_ues_load[0:original_associated_user_count[bs]], axis=None)

                    sorted_new_associated_ues_load = np.sort(associated_ues_load[original_associated_user_count[bs]:],
                                                             axis=None)
                    sorted_new_associated_ues_index = np.argsort(
                        associated_ues_load[original_associated_user_count[bs]:], axis=None)

                    sorted_associated_ues_load = np.concatenate((sorted_original_associated_ues_load,
                                                                 sorted_new_associated_ues_load))

                    sorted_associated_ues_index = np.concatenate((sorted_original_associated_ues_index,
                                                                  sorted_new_associated_ues_index +
                                                                  original_associated_user_count[bs]
                                                                  ))
                else:
                    sorted_associated_ues_load = np.sort(associated_ues_load, axis=None)
                    sorted_associated_ues_index = np.argsort(associated_ues_load, axis=None)

                sorted_served_ues_index, sorted_outage_ues = find_candidates(sorted_associated_ues_load,
                                                                             sorted_associated_ues_index, 1)
                sorted_served_ues_load = associated_ues_load[sorted_served_ues_index.astype(int)].copy()
                sorted_outage_ues_load = associated_ues_load[sorted_outage_ues.astype(int)].copy()
                served_ue_indicator[np.array(associated_ues)[sorted_served_ues_index.astype(int)].astype(int)] = 1
                if len(sorted_outage_ues) > 0:
                    actual_ue_load[np.array(associated_ues)[sorted_outage_ues.astype(int)].astype(int)] = (1 - np.sum(
                        sorted_served_ues_load)) * sorted_outage_ues_load / np.sum(sorted_outage_ues_load)

                actual_bs_load[bs] = np.sum(actual_ue_load[associated_ues])
            else:
                served_ue_indicator[associated_ues] = 1

            actual_ue_rate[associated_ues] = ue_packet_arrivals[associated_ues] * actual_ue_load[
                associated_ues] / associated_ues_load
            actual_bs_rate[bs] = np.sum(actual_ue_rate[associated_ues])
            server_ues_count[bs] = len(sorted_served_ues_index)
            dropped_ues_count[bs] = len(associated_ues) - server_ues_count[bs]
        else:
            server_ues_count[bs] = len(associated_ues)
            dropped_ues_count[bs] = len(associated_ues) - server_ues_count[bs]

    actual_mean_ue_rate = np.mean(actual_ue_rate[ue_rate_combined >= 0])
    fairness = (np.sum(actual_ue_rate)) ** 2 / (total_ues * np.sum(actual_ue_rate ** 2))
    return fairness, dropped_ues_count, actual_bs_load, actual_bs_rate, actual_ue_load, actual_ue_rate, served_ue_indicator

def find_candidates(sorted_associated_ues_load, sorted_associated_ues_index, free_load):
    selected_user = np.array([])
    rejected_user = np.array([])
    new_rejected_users = np.array([])
    new_selected_users = np.array([])
    user_length = len(sorted_associated_ues_load)

    for user in range(user_length):
        if sorted_associated_ues_load[user] <= free_load:
            selected_user = sorted_associated_ues_index[user]
            if user < user_length:
                new_free_load = free_load - sorted_associated_ues_load[user]
                remaining_users = list(range(user + 1, user_length))

                new_selected_users, new_rejected_users = find_candidates(sorted_associated_ues_load[remaining_users],
                                                                         sorted_associated_ues_index[remaining_users],
                                                                         new_free_load)

            selected_user = np.hstack((selected_user, new_selected_users))
            break
        rejected_user = np.hstack((rejected_user, sorted_associated_ues_index[user]))
    rejected_users = np.hstack((rejected_user, new_rejected_users))
    return selected_user, rejected_users

# Find outage ABSs
def find_outage_baseStations(user_served_indicator, user_association_current, user_association_previous):
    outage_indicator = np.zeros(total_base_stations)
    outage_counter = np.zeros(total_base_stations)
    index1 = np.where(np.array(user_served_indicator) == 0)[0]
    index2 = user_association_current[index1]  #
    index3 = np.where((user_association_current[index1]) == -1)[0]
    if len(index3) > 0:
        outage_indicator = np.ones(total_base_stations)
    else:
        outage_indicator[index2.astype(int)] = 1
    previous_serving_base_stations_of_dropped_users = user_association_previous[index1]
    previous_serving_base_stations_of_dropped_users = np.delete(previous_serving_base_stations_of_dropped_users,
                                                                np.where(
                                                                    previous_serving_base_stations_of_dropped_users == -1)[
                                                                    0], None)

    if len(previous_serving_base_stations_of_dropped_users) > 0:
        outage_indicator[previous_serving_base_stations_of_dropped_users.astype(int)] = 1
    return outage_indicator

# Calculate reward based on load and fairness
def utility_base_station(fairness, load):
    return torch.Tensor(0.5 * fairness + 0.5 * (1 - load))

# Normalization of UAVs' locations and transmit channels.
# Each location is normalized to a range of [0,1], and each channel is normalized using a one-hot encoder.
def state_standardization(cart_uavs, channel_uav):
    cart_uavs_normalized = cart_uavs.copy()
    cart_uavs_normalized[:, 0] = (cart_uavs[:, 0] + area_width_x / 2) / area_width_x
    cart_uavs_normalized[:, 1] = (cart_uavs[:, 1] + area_width_y / 2) / area_width_y
    cart_uavs_normalized[:, 2] = (cart_uavs[:, 2] - min_altitude_uav) / (max_altitude_uav - min_altitude_uav)
    channel_uav_one_hot = torch.nn.functional.one_hot(torch.tensor(channel_uav).to(torch.int64),
                                                      num_classes=total_subcarriers_ground)
    return cart_uavs_normalized, channel_uav_one_hot

# Optimize the policy network using a batch of transitions from the UAV's experience memory.
# The target network is used to compute the next state value.
def optimize_model(memory, Transition, policy_net, target_net, optimizer):
    if len(memory) < batch_size:
        return
    policy_net.train()
    target_net.train()
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    state_action_values = policy_net(state_batch.type(torch.float32).to(device)).gather(1,
                                                                             action_batch.type(torch.int64).to(device).unsqueeze(
                                                                                 -1))

    next_state_values = torch.zeros(batch_size, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states.type(torch.float32)).max(1)[0].detach()
    expected_state_action_values = (next_state_values * gamma) + reward_batch.to(device)

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    # loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# Calculate average over time
def calcuate_average(current_value, average_old, time_step):
    average_new = average_old.copy()
    if time_step == 0:
        average_new[:, time_step] = current_value
    else:
        average_new[:, time_step] = (current_value + (time_step) * average_old[:, time_step - 1]) / (time_step + 1)
    return average_new

