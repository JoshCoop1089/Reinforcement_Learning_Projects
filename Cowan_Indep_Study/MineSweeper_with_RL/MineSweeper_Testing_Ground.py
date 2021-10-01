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
import MineSweeper_TF_Functions as mstf
import MineSweeper_Network_Construction as msnc

import matplotlib.pyplot as plt
import seaborn as sns

##########################################################
##############ONLY CHANGE THESE!!!########################

# Board Specifics
dimension = 10
mine_percent = 0.4

# Learning Specifics
learning_param = 0.05
batch_fraction = 2          #Divide the states by this value to produce the batch

# Network Specifics  (check MineSweeper_Network_Construction.py for current layer setup)
filters = [32,64]
kernel_size = 3
dense_layer_nodes = [3*(dimension**2), dimension**2]
dropout_coef = 0.5
l2_val = 0.0001

# Training Specifics
num_episodes_per_update = 4000
num_training_times = 5

# Evaluation Specifics
num_games = 100

#########################################################
#################DON'T TOUCH BELOW#######################

num_mines = int(mine_percent * (dimension**2))
input_variables = (dimension, num_mines, learning_param, batch_fraction)
network_variables = (dropout_coef, l2_val, dense_layer_nodes)

# Establish baseline network
q_network_untrained = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
q_network_untrained.summary()

# Train the network, BE THE NETWORK
q_network_trained = msnc.create_base_network(filters, kernel_size, input_variables, network_variables)
q_network_trained = msnc.train_network(num_training_times, num_episodes_per_update, input_variables, q_network_trained)

# Can the network do the thing? Who knows?!!? Avg over a bunch of data
avg_clicks_random = []
avg_clicks_untrained = []
avg_clicks_trained = []

for x in range(1,num_games+1):
    print(f"Starting Game: {x} out of {num_games}")
    avg_click_random = mstf.play_one_game_random_choice_baseline(dimension, num_mines)
    avg_click_untrained = mstf.play_one_game_no_training_baseline(dimension, num_mines, q_network_untrained)
    avg_click_trained = mstf.play_one_game(dimension, num_mines, q_network_trained)
    
    # Data Aggregation
    avg_clicks_random.append(avg_click_random)
    avg_clicks_untrained.append(avg_click_untrained)
    avg_clicks_trained.append(avg_click_trained)
    
avg_percents_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]
avg_percents_untrained = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_untrained]
avg_percents_trained = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_trained]

# Plot the things!  KDE Plots for now
sns.displot([avg_clicks_random, avg_clicks_untrained, avg_clicks_trained], kind = 'kde', legend = False)
plt.title(f"Network Specifics -> Conv2D Filters: {filters} || Kernel Size: {kernel_size}" +
          f"\nDense Layers: {dense_layer_nodes} w/ ReLu || Dropout: {dropout_coef}  || L2 Reg = {l2_val}" +
          f"\nTraining -> Learning Rate: {learning_param} || Batch Updates: {num_training_times} " +
          f"|| Batch Size: {((dimension ** 2) * num_episodes_per_update)// batch_fraction}" +
          f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
          "\n"+ "- "*20 + f"\nRandom choice, untrained, and trained average scores over {num_games} games")
plt.legend(labels = ['Random Choice', 'Untrained', 'Trained'])
plt.xlabel('Game Score (higher is better!) [Percent of Optimal Moves Made]')
plt.ylabel('Density')