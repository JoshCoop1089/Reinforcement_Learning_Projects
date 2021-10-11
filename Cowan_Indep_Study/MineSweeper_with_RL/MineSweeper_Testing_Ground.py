# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:08:27 2021

@author: joshc
"""

"""
Evaluation Metrics for network playing a game:
    
Game Type: Lose on First Click
    Ratio of how many safe spaces are properly flagged vs total mines?
    Include info for misflagged mines?

Game Type: Play through to the end          
    Find the mean time when all mines have been clicked
        Keep track of which move clicks on a mine
        End of the game, find the average turn
        The higher the value, the more mines were clicked later in the game, 
            ie, the network made safe choices earlier, and was only left with mines later
                
    Using avg_mine_time:
        for a 10x10 board with 40 mines
            The worst possible game: (click all mines first)
                29.5
            The best possible game: (click all safe first)
                79.5
                
    Scaling function for scoring:
        safe_spaces = dim**2 - num_mines
        low = np.mean([x for x in range(0,safe_spaces)])
        high = np.mean([x for x in range(safe_spaces, dim**2)])
        percentile = (Score - Low)/(High - Low)
"""
import MineSweeper_Network_Construction as msnc
from MineSweeper_Base_Game import optimal_play_percent
from MineSweeper_Agent_Plays import play_one_game_random_choice_baseline
from MineSweeper_Agent_Plays import play_one_game_single_network
from MineSweeper_Agent_Plays import play_one_game_double_q

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##########################################################
##############ONLY CHANGE THESE!!!########################

# Board Specifics
dimension = 5
mine_percent = 0.4

# Learning Specifics
learning_param = 0.1
epsilon_greedy = 0.1

# Network Specifics  (check MineSweeper_Network_Construction.py for current layer setup)
filters = [64,128]
kernel_size = [3,3]
dense_layer_nodes = [5*(dimension**2), dimension**2]
dropout_coef = 0.25
l2_val = 0.0001

# Training Specifics (non fancy buffer) (only used for one version of regular q)
num_episodes_per_update = 1000
num_training_times_v1 = 5
batch_fraction = 1          #Divide the states by this value to produce the batch

# Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
games_per_buffer = 40
generations_per_buffer = 2
fits_per_generation = 25
num_buffer_refills = 3
num_training_times_q = 2
num_training_times_ac = 12

# Evaluation Specifics
num_games = 200

#########################################################
#################DON'T TOUCH BELOW#######################

num_mines = int(mine_percent * (dimension**2))
input_variables = (dimension, num_mines, learning_param, batch_fraction, epsilon_greedy)
network_variables = (dropout_coef, l2_val, dense_layer_nodes)
buffer_variables = (generations_per_buffer, games_per_buffer, fits_per_generation, num_buffer_refills)

# Establish baseline network
# q_network_untrained = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)

# # Regular Q networks
q_network_buffer_v1 = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
q_network_buffer_v2 = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
q_network_buffer_v1.summary()
q_network_buffer_v1, q_v1_avg_game_time = msnc.train_q_network_without_good_buffer(num_training_times_v1, num_episodes_per_update,
                                                                input_variables, q_network_buffer_v1)

q_networks = [q_network_buffer_v2]
q_network_buffer_v2, q_v2_avg_game_time = msnc.train_q_network_with_good_buffer(input_variables, buffer_variables,
                                                            num_training_times_q, q_networks)

# DoubleQ Networks
double_q_network_one = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
double_q_network_two = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
double_q_networks = [double_q_network_one, double_q_network_two]
double_q_networks, dq_avg_game_time = msnc.train_double_q_networks(input_variables, buffer_variables,
                                                 num_training_times_q, double_q_networks)

# ActorCritic Networks
actor_network = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
critic_network = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
actor_critic_networks = [actor_network, critic_network]
actor_critic_networks, ac_avg_game_time = msnc.train_actor_critic_networks(input_variables, buffer_variables,
                                                                            num_training_times_ac, actor_critic_networks)

# Can the network do the thing? Who knows?!!? Avg over a bunch of data
avg_clicks_random = []
avg_clicks_buffer_v1 = []
avg_clicks_buffer_v2 = []
avg_clicks_double = []
avg_clicks_actor_critic = []

for x in range(1,num_games+1):
    print(f"Starting Game: {x} out of {num_games}")
    avg_click_random = play_one_game_random_choice_baseline(dimension, num_mines)
    avg_click_buffer_v1 = play_one_game_single_network(dimension, num_mines, q_network_buffer_v1)
    avg_click_buffer_v2 = play_one_game_single_network(dimension, num_mines, q_network_buffer_v2)
    avg_click_double = play_one_game_double_q(dimension, num_mines, double_q_networks)
    avg_click_actor_critic = play_one_game_single_network(dimension, num_mines, actor_critic_networks[1])
    
    # Data Aggregation
    avg_clicks_random.append(avg_click_random)
    avg_clicks_buffer_v1.append(avg_click_buffer_v1)
    avg_clicks_buffer_v2.append(avg_click_buffer_v2)
    avg_clicks_double.append(avg_click_double)
    avg_clicks_actor_critic.append(avg_click_actor_critic)
    
avg_score_random = [100*optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]
avg_score_buffer_v1 = [100*optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_buffer_v1]
avg_score_buffer_v2 = [100*optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_buffer_v2]
avg_score_double = [100*optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_double]
avg_score_actor_critic = [100*optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_actor_critic]


display_data = []
display_data.append(avg_score_random)
display_data.append(avg_score_buffer_v1)
display_data.append(avg_score_buffer_v2)
display_data.append(avg_score_double)
display_data.append(avg_score_actor_critic)
mean_scores = [round(np.mean(score), 1) for score in display_data]

# Plot the things!
plt.hist(display_data, color=['r','b','g', 'y', 'k'], alpha=0.5)
plt.title(f"Network Specifics -> Conv2D Filters: {filters} || Kernel Size: {kernel_size}" +
          f"\nDense Layers: {dense_layer_nodes} w/ ReLu || Dropout: {dropout_coef}  || L2 Reg = {l2_val}" +
          f"\nTraining -> Learning Rate: {learning_param}" + 
          f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
          "\n"+ "- "*20 + 
          f"\nRandom choice and trained average scores over {num_games} games" +
          f"\nRandom: {mean_scores[0]}/100 || Buffer v1: {mean_scores[1]}/100, {q_v1_avg_game_time} s " + 
              f"|| Buffer v2: {mean_scores[2]}/100, {q_v2_avg_game_time} s" + 
          f"\nDoubleQ: {mean_scores[3]}/100, {dq_avg_game_time} s || " + 
              f"ActorCritic: {mean_scores[4]}/100, {ac_avg_game_time} s")

plt.legend(labels = ['Random Choice', 'Q Buffer v1', "Q Buffer v2", "DoubleQ", "ActorCritic"])
plt.xlabel('Game Score (higher is better!) [Percent of Optimal Moves Made]')
plt.show()


sns.displot(display_data, kind = 'kde', legend = False)
plt.title(f"Network Specifics -> Conv2D Filters: {filters} || Kernel Size: {kernel_size}" +
          f"\nDense Layers: {dense_layer_nodes} w/ ReLu || Dropout: {dropout_coef}  || L2 Reg = {l2_val}" +
          f"\nTraining -> Learning Rate: {learning_param}" + 
          f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
          "\n"+ "- "*20 + 
          f"\nRandom choice and trained average scores over {num_games} games" +
          f"\nRandom: {mean_scores[0]}/100 || Buffer v1: {mean_scores[1]}/100, {q_v1_avg_game_time} s" + 
              f"|| Buffer v2: {mean_scores[2]}/100, {q_v2_avg_game_time} s " + 
          f"\nDoubleQ: {mean_scores[3]}/100, {dq_avg_game_time} s || " + 
              f"ActorCritic: {mean_scores[4]}/100, {ac_avg_game_time} s")
plt.legend(labels = ['Random Choice', 'Q Buffer v1', "Q Buffer v2", "DoubleQ", "ActorCritic"])
plt.xlabel('Game Score (higher is better!) [Percent of Optimal Moves Made]')
plt.ylabel('Density')
plt.show()