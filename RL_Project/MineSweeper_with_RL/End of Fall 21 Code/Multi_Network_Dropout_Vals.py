# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:01:37 2021

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
import random

import MineSweeper_Base_Game as ms
from predict_mines import generate_test_data
from Minesweeper_Non_Perfect_Info import play_one_game_single_network
from Minesweeper_Non_Perfect_Info import play_one_game_random_choice_baseline
from Minesweeper_Non_Perfect_Info import train_q_network_v1
from Minesweeper_Non_Perfect_Info import one_hot_encode_next_state


def make_all_models(dimension, size, scale, dropout_coef, activator_func):
    input_shape = (dimension, dimension, 11)
    network_v1_no_drop = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[0]*dropout_coef),
                        layers.Dense(size[0]*(dimension**2)),
                        layers.Dropout(scale[0]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v1_drop = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[0]*dropout_coef),
                        layers.Dense(size[1]*(dimension**2)),
                        layers.Dropout(scale[0]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v1_regularized = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[0]*dropout_coef),
                        layers.Dense(size[2]*(dimension**2)),
                        layers.Dropout(scale[0]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v2_no_drop = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[1]*dropout_coef),
                        layers.Dense(size[0]*(dimension**2)),
                        layers.Dropout(scale[1]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
                        
    network_v2_drop = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[1]*dropout_coef),
                        layers.Dense(size[1]*(dimension**2)),
                        layers.Dropout(scale[1]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v2_regularized = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[1]*dropout_coef),
                        layers.Dense(size[2]*(dimension**2)),
                        layers.Dropout(scale[1]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v3_no_drop = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[2]*dropout_coef),
                        layers.Dense(size[0]*(dimension**2)),
                        layers.Dropout(scale[2]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v3_drop = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[2]*dropout_coef),
                        layers.Dense(size[1]*(dimension**2)),
                        layers.Dropout(scale[2]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    network_v3_regularized = tf.keras.Sequential([
                        tf.keras.Input(shape = input_shape),
                        layers.Flatten(),
                        layers.Dropout(scale[2]*dropout_coef),
                        layers.Dense(size[2]*(dimension**2)),
                        layers.Dropout(scale[2]*dropout_coef),
                        layers.Dense(dimension**2, activation = activator_func),
                        layers.Reshape(target_shape=(dimension, dimension))
        ])
    models = [network_v1_no_drop, network_v1_drop, network_v1_regularized, network_v2_no_drop, network_v2_drop, network_v2_regularized,
                          network_v3_no_drop, network_v3_drop, network_v3_regularized]

    return models

def get_network_play_results (num_games, dimension, num_mines, network_name, version_name, network_type):
    avg_clicks = []    
    vals = [x*num_games//10 for x in range (11)]
    for x in range(1,num_games+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games} for network: {version_name} w/ {network_type}")
        avg_clicks.append(play_one_game_single_network(dimension, num_mines, network_name))
    
    avg_score = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks]
    return avg_score 

def freq_dict(inputs):
    freq_dic = {}
    for val in inputs:
        freq_dic[val] = freq_dic.get(val,0)+1
    return freq_dic

def distinct_vals_b_minus_a(a, b):
    a_freq = freq_dict(a)
    b_freq = freq_dict(b)
    abkeys = set([*a_freq, *b_freq])
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

def train_and_eval_mine_predictions(dimension, num_mines, num_games_train, 
                                    num_training_rounds, num_games_eval, mine_network):
    input_shape = (dimension, dimension, 11)

    checkpoint_filepath = '\model_checkpoints\checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=checkpoint_filepath,
                                    save_weights_only=True,
                                    monitor='loss',
                                    mode='min',
                                    save_best_only=True)

    loss = []
    accuracy = []
    for _ in range(num_training_rounds):    
        state = []
        label = []
        history = generate_test_data(dimension, num_mines, num_games_train)  
        batch = random.sample(history.items(), 20*(dimension**2))
    
        for k, (s,l) in batch:
            state.append(s)
            label.append(l)
        states = tf.convert_to_tensor(state)
        labels = tf.convert_to_tensor(label)
        outputs = mine_network.fit(states, labels, callbacks = [model_checkpoint_callback])
        loss.extend(outputs.history['loss'])
        accuracy.extend(outputs.history['accuracy'])
        
    mine_network.load_weights(checkpoint_filepath)       
    metrics = [loss, accuracy]
    avg_score = []
    avg_times = []
    
    for game in range(num_games_eval):
        if game in [num_games_eval//4 -1, 2*num_games_eval//4 -1 , 
                                3*num_games_eval//4-1, num_games_eval -1]:
            print(f"Starting Evaluation Game {game + 1} out of {num_games_eval}")
        board = ms.make_board(dimension, num_mines)
        b_enc = np.zeros(input_shape)
        count = 0
        mine_times = []
        
        while count < dimension ** 2:
            b_enc_t = tf.convert_to_tensor(np.expand_dims(b_enc, axis=0))
            pred = mine_network.predict(b_enc_t)
            locs = np.where(board[:,:,0] == 0)
            places = list(zip(locs[0], locs[1]))
            actions = [pred[0][x][y] for (x,y) in places]
        
            next_loc = places[np.argmin(actions)]
            if board[next_loc[0]][next_loc[1]][1] == 1:
                mine_times.append(count)
            b_enc = one_hot_encode_next_state(b_enc, board, next_loc)
            count += 1
        
        avg_times.append(np.mean(mine_times))
    
    avg_score = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_times]
    return avg_score, metrics

################################################################
# Board Specifics
dimension = 3
mine_percent = 0.4

# Learning Specifics
learning_param = 0.05
epsilon_greedy = 0.05


# # -----> Testing Outputs Quickly <----- #
# # Training Specifics (non fancy buffer) (only used for one version of regular q)
# num_episodes_per_update = 5
# num_training_times_v1 = 2
# batch_fraction = 1          #Divide the states by this value to produce the batch

# # Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
# games_per_buffer = 5
# generations_per_buffer = 1
# fits_per_generation = 10
# num_buffer_refills = 1
# num_training_times_q = 1

# num_games_train_mines = 20
# num_training_times_mines = 1

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
num_training_times_q = 1

# Mine Prediction Coefs
num_games_train_mines = 1000
num_training_times_mines = 100

dense_layer_size = 80
size = [0.75*dense_layer_size, dense_layer_size, 1.25*dense_layer_size]

dropout_coef = 0.375
scale = [0.75, 1, 1.25]
    
# Evaluation Specifics
num_games = 1000


##############################################################
version = scale
network_style = size
input_shape = (dimension, dimension, 11)
num_mines = int(mine_percent * (dimension**2))
input_variables = (dimension, num_mines, learning_param, batch_fraction, epsilon_greedy)
buffer_variables = (generations_per_buffer, games_per_buffer, fits_per_generation, num_buffer_refills)

# Set up the random baseline
avg_clicks_random = []    
vals = [x*num_games//10 for x in range (11)]
for x in range(1,num_games+1):
    if x in vals:
        print(f"Starting Game: {x} out of {num_games} for network: Random Play")
    avg_clicks_random.append(play_one_game_random_choice_baseline(dimension, num_mines))
avg_score_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]

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
              f"\nDrop Coef: {dropout_coef} || Base Dense Size: {dense_layer_size}"+"\n"+ "- "*20 + "\nBlue: Random, Orange: Trained")
fig1.suptitle(f"Score Distributions for {num_games} games\nBlue: Random, Orange: Trained")
fig2.suptitle("Loss over Training")
fig3.suptitle(f"Score Results\nRandom Choice: {round(np.mean(avg_score_random), 1)}/100 || Median: {round(np.median(avg_score_random), 1)}")
fig.tight_layout()  
fig1.tight_layout()  
fig2.tight_layout()  
fig3.tight_layout()  


        
# # Testing Mine Predictor
# activator_func = 'sigmoid'

# Testing Q_Learning
activator_func = 'relu'

models = make_all_models(dimension, size, scale, dropout_coef, activator_func)

# Iterate over the models
for i in range(3):
    for j in range(3):
        version_name = str(version[i])
        network_type = str(network_style[j])
        print("\n\n ---> Starting New Network Type <---\n\t", version_name, network_type)
        model = models[3*i+j]
        # model.summary()    
        
        # Use untrained model to play game
        # untrained_results = get_network_play_results(num_games, dimension, num_mines, model, version_name, network_type)
        
        # Train model
        # model, training_loss, training_accuracy = train_model_buffer_v2()
           
        
        ###############################################################
        ###### Only have one of these active at a time for now!!! #####
        
        # Q-Learning Models
        model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
        model, _, [training_loss, training_accuracy] = train_q_network_v1(input_variables, num_training_times_v1, model, 
                                                                          num_episodes_per_update, fits_per_generation)
        # Use trained model to play game
        trained_results = get_network_play_results(num_games, dimension, num_mines, model, version_name, network_type)
        

        # # Mine prediction Models
        # model.compile(optimizer='adam',
        #                   loss=tf.keras.losses.CategoricalCrossentropy(),
        #                   metrics=['accuracy'])
        # trained_results, [training_loss, training_accuracy] = train_and_eval_mine_predictions(dimension, num_mines, num_games_train_mines, 
        #                                                   num_training_times_mines, num_games, model)
        
        
        ###############################################################
        
        # MAKE THE UBERGRAPH UNLEASH THE KRAKEN
        display_data = []
        display_data.append(avg_score_random)
        # display_data.append(untrained_results)
        display_data.append(trained_results)
        
        # Histograms!
        a_out, b_out = distinct_vals_b_minus_a(avg_score_random, trained_results)      
        ax[i,j].bar(a_out.keys(), a_out.values())
        ax[i,j].bar(b_out.keys(), b_out.values())
        # plt.show()
        # ax[i,j].hist(display_data, color = ['b', 'r', 'g'])
        
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