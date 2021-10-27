# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 13:29:03 2021

@author: joshc
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 18:01:38 2021

@author: joshc
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import MineSweeper_Base_Game as ms
from Minesweeper_Non_Perfect_Info import play_one_game_single_network
from Minesweeper_Non_Perfect_Info import play_one_game_random_choice_baseline
from Minesweeper_Non_Perfect_Info import train_q_network_v1



################################################################
# Board Specifics
dimension = 3
mine_percent = 0.4

# Learning Specifics
learning_param = 0.05
epsilon_greedy = 0.05
l2_vl = 0.025
l2_val = [round(0.75*l2_vl, 5), l2_vl, 1.25*l2_vl]

# # -----> Testing Outputs Quickly <----- #
# # Training Specifics (non fancy buffer) (only used for one version of regular q)
# num_episodes_per_update = 20
# num_training_times_v1 = 2
# batch_fraction = 1          #Divide the states by this value to produce the batch

# # Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
# games_per_buffer = 5
# generations_per_buffer = 1
# fits_per_generation = 10
# num_buffer_refills = 1
# num_training_times_q = 1

# # Evaluation Specifics
# num_games = 5


# -----> Actual Data Collection Values <-----  #
# Training Specifics (non fancy buffer) (only used for one version of regular q)
num_episodes_per_update = 100
num_training_times_v1 = 8
batch_fraction = 1          #Divide the states by this value to produce the batch

# Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
games_per_buffer = 100
generations_per_buffer = 2
fits_per_generation = 20
num_buffer_refills = 3
num_training_times_q = 2
    
# Evaluation Specifics
num_games = 1000

version = ["One Large Dense", "One Small Dense", "Two Small Dense"]
network_style = l2_val

##############################################################


num_mines = int(mine_percent * (dimension**2))
input_variables = (dimension, num_mines, learning_param, batch_fraction, epsilon_greedy)
buffer_variables = (generations_per_buffer, games_per_buffer, fits_per_generation, num_buffer_refills)

input_shape = (dimension, dimension, 11)


network_v1_no_drop = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(100*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v1_drop = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(100*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v1_regularized = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(100*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v2_no_drop = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(50*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v2_drop = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(50*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v2_regularized = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(50*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v3_no_drop = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(25*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Dense(25*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[0])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v3_drop = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(25*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Dense(25*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[1])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
network_v3_regularized = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dense(25*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Dense(25*(dimension**2), kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Dense(dimension**2, kernel_regularizer=regularizers.l2(l2_val[2])),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])


def get_network_play_results (num_games, dimension, num_mines, network_name, version_name, network_type):
    avg_clicks = []    
    vals = [x*num_games//10 for x in range (11)]
    for x in range(1,num_games+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games} for network: {version_name} w/ {network_type}")
        avg_clicks.append(play_one_game_single_network(dimension, num_mines, network_name))
    
    avg_score = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks]
    return avg_score    

# Set up the random baseline
avg_clicks_random = []    
vals = [x*num_games//10 for x in range (11)]
for x in range(1,num_games+1):
    if x in vals:
        print(f"Starting Game: {x} out of {num_games} for network: Random Play")
    avg_clicks_random.append(play_one_game_random_choice_baseline(dimension, num_mines))
avg_score_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]



models = [network_v1_no_drop, network_v1_drop, network_v1_regularized, network_v2_no_drop, network_v2_drop, network_v2_regularized,
                      network_v3_no_drop, network_v3_drop, network_v3_regularized]

# So many graphs to set up
fig, ax = plt.subplots(3,3, sharex='col', sharey = 'row', figsize=(10,10)) 
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig1, ax1 = plt.subplots(3,3, sharex='col', sharey = 'row', figsize=(10,10)) 
fig1.subplots_adjust(hspace=0.4, wspace=0.5)
fig2, ax2 = plt.subplots(3,3, figsize=(10,10))   
fig2.subplots_adjust(hspace=0.5, wspace=0.5)  
fig3, ax3 = plt.subplots(3,3, figsize=(10,10))   
fig3.subplots_adjust(hspace=0.5, wspace=0.5) 

for axe, col in zip(ax[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax[:,0], version):
    axe.set_ylabel(row, rotation=90, size='large')

for axe, col in zip(ax1[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax1[:,0], version):
    axe.set_ylabel(row, rotation=90, size='large')
    
for axe, col in zip(ax2[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax2[:,0], version):
    axe.set_ylabel(row, rotation=90, size='large')

for axe, col in zip(ax3[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax3[:,0], version):
    axe.set_ylabel(row, rotation=90, size='large')
    
fig.suptitle(f"Unique Scores from Random and Trained for {num_games} games\nBoard Size: {dimension}x{dimension} || Mines: {num_mines}" + 
              "\n"+ "- "*20 + "\nBlue: Random, Orange: Trained")
fig1.suptitle(f"Score Distributions for {num_games} games\nBlue: Random, Orange: Trained")
fig2.suptitle("Loss over Training")
fig3.suptitle(f"Score Results\nRandom Choice: {round(np.mean(avg_score_random), 1)}/100 || Median: {round(np.median(avg_score_random), 1)}")
fig.tight_layout()  
fig1.tight_layout()  
fig2.tight_layout()  
fig3.tight_layout()  

def freq_dict(inputs):
    freq_dic = {}
    for val in inputs:
        freq_dic[val] = freq_dic.get(val,0)+1
    return freq_dic

def distinct_vals_b_minus_a(a, b):
    a_freq = freq_dict(a)
    b_freq = freq_dict(b)
    
    abkeys = set([*a_freq, *b_freq])
    # print(abkeys)
    bmina = {}
    for key in abkeys:
        if key in b_freq.keys() and key in a_freq.keys():
            bmina[key] = b_freq[key]-a_freq[key]
        elif key in b_freq.keys():
            bmina[key] = b_freq[key]
        elif key in a_freq.keys():
            bmina[key] = -1*a_freq[key]
            
    a_out = {}
    b_out = {}
    for key in bmina:
        if bmina[key] > 0:
            b_out[key] = bmina[key]
        elif bmina[key] < 0:
            a_out[key] = -1*bmina[key]
            
    return a_out, b_out

# Iterate over the models
for i in range(3):
    for j in range(3):
        version_name = version[i]
        network_type = str(network_style[j])
        print("\n\n ---> Starting New Network Type <---\n\t", version_name, network_type)
        model = models[3*i+j]
        # model.summary()    
        model.compile(optimizer='adam',
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=['accuracy'])
        
        # Use untrained model to play game
        # untrained_results = get_network_play_results(num_games, dimension, num_mines, model, version_name, network_type)
        
        # Q-Learning Models
        model, _, [training_loss, training_accuracy] = train_q_network_v1(input_variables, num_training_times_v1, model, 
                                                                          num_episodes_per_update, fits_per_generation)
        # Use trained model to play game
        trained_results = get_network_play_results(num_games, dimension, num_mines, model, version_name, network_type)
        

        # MAKE THE UBERGRAPH UNLEASH THE KRAKEN
        display_data = []
        display_data.append(avg_score_random)
        # display_data.append(untrained_results)
        display_data.append(trained_results)
        
        # Unique Score Results!
        a_out, b_out = distinct_vals_b_minus_a(avg_score_random, trained_results)      
        ax[i,j].bar(a_out.keys(), a_out.values())
        ax[i,j].bar(b_out.keys(), b_out.values())
        
        # KDE's! But Double! And Vertical! Called Violins! With Quartiles!
        sns.violinplot(data = display_data, legend = False, ax = ax1[i,j], inner = 'quartile')
        
        # Training Loss Over Time!
        epochs = range(1, len(training_loss) + 1)
        ax2[i,j].plot(epochs, training_loss, 'bo')
        
        # Numbers!
        results = str(
            # f"\n\nUn:{round(np.mean(untrained_results), 1)} || M:{round(np.median(untrained_results), 1)}" + 
                      f"\nTr: {round(np.mean(trained_results), 1)}  || M:{round(np.median(trained_results), 1)}")
        ax3[i, j].text(0.5, 0.5, results,
                      fontsize=14, ha='center')
        
fig.savefig('scorediff.png')
fig1.savefig('violins.png')
fig2.savefig('loss.png')
fig3.savefig('meanvals.png')
