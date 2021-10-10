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
import MineSweeper_Base_Game as ms
import MineSweeper_TF_Functions_Regular_Q as rq
import MineSweeper_Network_Construction as msnc

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

##########################################################
##############ONLY CHANGE THESE!!!########################

# Board Specifics
dimension = 8
mine_percent = 0.4

# Learning Specifics
learning_param = 0.075
epsilon_greedy = 0.1

# Network Specifics  (check MineSweeper_Network_Construction.py for current layer setup)
filters = [32,64]
kernel_size = [3,3]
dense_layer_nodes = [3*(dimension**2), dimension**2]
dropout_coef = 0.5
l2_val = 0.0001

# Training Specifics (non fancy buffer) (only used for one version of regular q)
num_episodes_per_update = 1000
num_training_times = 5
batch_fraction = 1          #Divide the states by this value to produce the batch

# Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
generations_per_buffer = 3
games_per_buffer = 60
fits_per_buffer_fill = 3

# Evaluation Specifics
num_games = 100

#########################################################
#################DON'T TOUCH BELOW#######################

num_mines = int(mine_percent * (dimension**2))
input_variables = (dimension, num_mines, learning_param, batch_fraction, epsilon_greedy)
network_variables = (dropout_coef, l2_val, dense_layer_nodes)
buffer_variables = (generations_per_buffer, games_per_buffer, fits_per_buffer_fill)

# Establish baseline network
# q_network_untrained = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)

# Regular Q network
q_network_buffer_v1 = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
q_network_buffer_v2 = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
q_network_buffer_v1.summary()
q_network_buffer_v1 = msnc.train_q_network_without_good_buffer(num_training_times, num_episodes_per_update,
                                                               input_variables, q_network_buffer_v1)

q_networks = [q_network_buffer_v2]
q_network_buffer_v2 = msnc.train_q_network_with_good_buffer(input_variables, buffer_variables,
                                                                            num_training_times, q_networks)

# # DoubleQ Networks
# update_type = "DoubleQ"
# double_q_network_one = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
# double_q_network_two = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
# double_q_networks = [double_q_network_one, double_q_network_two]
# double_q_network_one, double_q_network_two = msnc.train_double_q_networks()

# # ActorCritic Networks (to be filled in later)
# update_type = "ActorCritic"
# actor_network = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
# critic_network = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
# actor_critic_networks = [actor_network, critic_network]
# actor_network, critic_network = msnc.train_actor_critic_networks()

# Can the network do the thing? Who knows?!!? Avg over a bunch of data
avg_clicks_random = []
avg_clicks_buffer_v1 = []
avg_clicks_buffer_v2 = []

for x in range(1,num_games+1):
    print(f"Starting Game: {x} out of {num_games}")
    avg_click_random = rq.play_one_game_random_choice_baseline(dimension, num_mines)
    avg_click_buffer_v1 = rq.play_one_game_single_q(dimension, num_mines, q_network_buffer_v1)
    avg_click_buffer_v2 = rq.play_one_game_single_q(dimension, num_mines, q_network_buffer_v2)
    
    # Data Aggregation
    avg_clicks_random.append(avg_click_random)
    avg_clicks_buffer_v1.append(avg_click_buffer_v1)
    avg_clicks_buffer_v2.append(avg_click_buffer_v2)
    
avg_score_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]
avg_score_buffer_v1 = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_buffer_v1]
avg_score_buffer_v2 = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_buffer_v2]


display_data = []
display_data.append(avg_score_random)
display_data.append(avg_score_buffer_v1)
display_data.append(avg_score_buffer_v2)
mean_scores = [round(np.mean(score), 1) for score in display_data]

# Plot the things!
plt.hist(display_data, color=['r','b', 'g'], alpha=0.5)
plt.title(f"Network Specifics -> Conv2D Filters: {filters} || Kernel Size: {kernel_size}" +
          f"\nDense Layers: {dense_layer_nodes} w/ ReLu || Dropout: {dropout_coef}  || L2 Reg = {l2_val}" +
          f"\nTraining -> Learning Rate: {learning_param} || Batch Updates: {num_training_times} " +
                  f"|| Batch Size: {((dimension ** 2) * num_episodes_per_update)// batch_fraction}" +
          f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
          "\n"+ "- "*20 + 
          f"\nRandom choice and trained average scores over {num_games} games" +
          f"\nRandom Avg Score: {mean_scores[0]}/100 || Buffer v1 Avg Score: " + 
              f"{mean_scores[1]}/100 || Buffer v2 Avg Score: {mean_scores[2]}/100")

plt.legend(labels = ['Random Choice', 'Q Buffer v1', "Q Buffer v2"])
plt.xlabel('Game Score (higher is better!) [Percent of Optimal Moves Made]')
plt.show()


sns.displot(display_data, kind = 'kde', legend = False)
plt.title(f"Network Specifics -> Conv2D Filters: {filters} || Kernel Size: {kernel_size}" +
          f"\nDense Layers: {dense_layer_nodes} w/ ReLu || Dropout: {dropout_coef}  || L2 Reg = {l2_val}" +
          f"\nTraining -> Learning Rate: {learning_param} || Batch Updates: {num_training_times} " +
                  f"|| Batch Size: {((dimension ** 2) * num_episodes_per_update)// batch_fraction}" +
          f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
          "\n"+ "- "*20 + 
          f"\nRandom choice and trained average scores over {num_games} games" +
          f"\nRandom Avg Score: {mean_scores[0]}/100 || Buffer v1 Avg Score: " + 
              f"{mean_scores[1]}/100 || Buffer v2 Avg Score: {mean_scores[2]}/100")
plt.legend(labels = ['Random Choice', 'Q Buffer v1', "Q Buffer v2"])
plt.xlabel('Game Score (higher is better!) [Percent of Optimal Moves Made]')
plt.ylabel('Density')
plt.show()