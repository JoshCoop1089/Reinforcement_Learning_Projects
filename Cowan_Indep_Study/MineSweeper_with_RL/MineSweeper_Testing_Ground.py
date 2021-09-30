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
        Avg number of safe space clicked before mine?
            How to deal with the end of the game when there are minimal safe spaces?
            
        *** Find the mean time when all mines have been clicked
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
            
            Percentile = (Score - Low)/(High - Low)
"""
import MineSweeper_Base_Game as ms
import MineSweeper_TF_Functions as mstf
import MineSweeper_Network_Construction as msnc

import matplotlib.pyplot as plt
import seaborn as sns

# Board Specifics
dimension = 10
num_mines = 40

# Learning Specifics
learning_param = 0.01
batch_fraction = 2
num_episodes_per_update = 1000
input_variables = (dimension, num_mines, learning_param, batch_fraction, num_episodes_per_update)

# Network Specifics
filters = 32
kernel_size = (3,3)
num_training_times = 20

# Make the Network, train the network, BE THE NETWORK
q_network = msnc.train_network(filters, kernel_size, num_training_times, input_variables)

def avg_time_per_game(avg, new_game_time, game_num):
    new_avg = (avg * (game_num - 1) + new_game_time)/game_num
    return round(new_avg, 3)

# Can the network do the thing? Who knows?!!? Avg over a bunch of data
num_games = 100
avg_clicks = []
avg_time = 0

for x in range(1,num_games+1):
    print(f"Starting Game: {x} out of {num_games}")
    avg_click_time = mstf.play_one_game(dimension, num_mines, q_network)
    avg_clicks.append(avg_click_time)
    
avg_percents = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks]
sns.displot(x = avg_percents, kind = 'kde')
plt.title("Network Specifics:"
          f"\nFilter Size: {filters} || Kernel Size: {kernel_size} || Learning Parameter: {learning_param}" + 
          f"\nBatch Updates: {num_training_times} || Batch Size: {((dimension ** 2) * num_episodes_per_update)// batch_fraction} state/qboard pairs" +
          "\n"+ "- "*20 + f"\nAfter training, average scores over {num_games} games")
plt.xlabel('Percent of Optimal Moves Made')
plt.ylabel('Density')
