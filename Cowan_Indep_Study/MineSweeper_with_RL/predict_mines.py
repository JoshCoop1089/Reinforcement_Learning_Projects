# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 11:00:36 2021

@author: joshc

Mine Prediction Sub Project:
    
    Inputs:
        Single Board state
        Only visible locations
        Use one hot encoder to indicate click, mine, clue number
    Outputs:
        Full board of mine locations
    
    Network Setup:
        Input Layer
        Single Dense Layer (size? just use 4*dim**2 because?)
        Single Dense Output of size dim**2
        Reshape Layer into dimxdim

"""
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers

import numpy as np
import copy, random

import MineSweeper_Base_Game as ms


def one_hot_encode_next_state(board_enc, ref_board, next_loc, playing = False):
    
    if not playing:
        board = copy.deepcopy(board_enc) 
    else:
        board = board_enc
        
    temp_encoder = np.zeros((11))
    x,y = next_loc
    
    # Encode the click
    temp_encoder[0] = 1
    
    # If the location is mined
    if ref_board[x][y][1] == 1:
        temp_encoder[1] = 1
        
    # Encode the clue value of the square
    else:
        loc = int(ref_board[x][y][2] * 8) + 2
        temp_encoder[loc] = 1
        
    ref_board[x][y][0] = 1
    board[x][y] = temp_encoder
    return board

def generate_test_data(dimension, num_mines, num_games):
    state_counter = 0
    history = {}
    for _ in range(num_games):
        board = ms.make_board(dimension, num_mines)
        # ms.print_board(board, full_print=True)
        mine_board = board[:,:,1]
        state = np.zeros((dimension, dimension, 11))
        # print(mine_board)
        game_over = False
        
        while not game_over:
            # Generate random next board state
            # Find all available locations
            locs = np.where(board[:,:,0] == 0)
            places = list(zip(locs[0], locs[1]))
            
            if len(places) == 0:
                game_over = True
            else:
                # ms.print_board(board)
                # print(state)
                history[state_counter] = (state, mine_board)
                # Random Choose from list
                next_loc = random.choice(places) 
                state = one_hot_encode_next_state(state, board, next_loc)  
                state_counter += 1
    return history


if __name__ == '__main__':
    dimension = 5
    mine_percent = 0.4
    num_games = 2000
    
    num_training_rounds = 50
    num_games_eval = 500
    
    dropout = 0.37
    dense_size = 100*(dimension**2)
    
    
    # ------------------- #
    
    
    num_mines = int(mine_percent * (dimension**2))
    input_shape = (dimension, dimension, 11)
    
    mine_network = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dropout(dropout),
                    layers.Dense(dense_size),
                    layers.Dropout(dropout),
                    # layers.Dense(dense_size),
                    layers.Dense(dimension**2, activation = 'sigmoid'),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
    mine_network.summary()
    
    mine_network.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])              
    for _ in range(num_training_rounds):    
        state = []
        label = []
        history = generate_test_data(dimension, num_mines, num_games)  
        batch = random.sample(history.items(), 20*(dimension**2))
    
        # print(type(history))
        for k, (s,l) in batch:
            state.append(s)
            label.append(l)
        states = tf.convert_to_tensor(state)
        labels = tf.convert_to_tensor(label)
        outputs = mine_network.fit(states, labels)
    
    
    # full = tf.convert_to_tensor(np.expand_dims(ms.make_board(dimension, 100), axis = 0)) 
    # full = np.zeros((1,dimension, dimension, 11))
    # full[:,:,:,0] = 1
    # full[:,:,:,1] = 1
    # print(mine_network.predict(full))
    
    avg_score = []
    avg_times = []
    
    # print(history)
    for game in range(num_games_eval):
        if game in [num_games_eval//4 -1, 2*num_games_eval//4 -1 , 
                                3*num_games_eval//4-1, num_games_eval -1]:
            print(f"Starting Evaluation Game {game + 1} out of {num_games_eval}")
        board = ms.make_board(dimension, num_mines)
        # ms.print_board(board, full_print=True)
        b_enc = np.zeros(input_shape)
        count = 0
        mine_times = []
        
        while count < dimension ** 2:
            b_enc_t = tf.convert_to_tensor(np.expand_dims(b_enc, axis=0))
            pred = mine_network.predict(b_enc_t)
            # print(pred)
            locs = np.where(board[:,:,0] == 0)
            places = list(zip(locs[0], locs[1]))
            actions = [pred[0][x][y] for (x,y) in places]
        
            next_loc = places[np.argmin(actions)]
            # print(next_loc)
            if board[next_loc[0]][next_loc[1]][1] == 1:
                mine_times.append(count)
            b_enc = one_hot_encode_next_state(b_enc, board, next_loc)
            # ms.print_board(board, full_print=False)
            count += 1
        
        avg_times.append(np.mean(mine_times))
    
    avg_score = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_times]
    plt.hist(avg_score)
    plt.title(f"Board Size: {dimension}x{dimension} || Num Mines: {num_mines}" +
              f"\nDropout Coef: {dropout} || Hidden Dense Layer: {dense_size} nodes"
              f"\nAvg Score over {num_games_eval} games: {round(np.mean(avg_score), 1)}/100" + 
              f"\nMedian Score: {round(np.median(avg_score), 1)} || Standard Deviation: {round(np.std(avg_score), 1)}")
    plt.xlabel('Game Score (higher is better!) [Percent of Optimal Moves Made]')
    plt.show()
    
    sns.violinplot(data = avg_score, inner = 'quartile')
    plt.ylabel('Game Score [Percent of Optimal Moves Made]')
