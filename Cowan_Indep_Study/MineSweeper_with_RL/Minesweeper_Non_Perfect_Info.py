# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:07:29 2021

@author: joshc
"""

"""
OneHot Encoded Non Perfect Info Main Game Tests

Define Params
Build Network
Generate Training Data
Generate Eval Data
Output Graphs


"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

import MineSweeper_Base_Game as ms


import time
import random
import copy


def q_value_update(state1_q_board, state2_q_board, reward_board, learning_param):
    """
    state1_q_board: holds the q values for the current state of the board
    state2_q_board: holds the q values for the next state of the board
    reward_board: gives the rewards for every space in the board state
    """
    # no discounting?
    gamma = 1
    
    update_q_board = np.zeros(state1_q_board.shape)    
    q_new = reward_board + gamma * np.amax(state2_q_board)
    update_q_board = state1_q_board + learning_param * (q_new - state1_q_board)
    return update_q_board

def get_greedy_next_action(q_board, board, epsilon):
    """
    The zipping on list_of_locs might need to be changed when flag functionality is added?
    """
    action_is_flag = False
    next_action_loc = (0,0)
        
    # This is such a janky way of doing this wtf lol
    # Get list of all available spots
    locs = np.where(board == 0)
    places = list(zip(locs[0], locs[1], locs[2]))
    new = [(x,y) for (x,y,z) in places if z == 0]
    if len(new) == 0:
        pass
        
    # Get location of highest q in available locations
    else:
        # q_board is a 1xdimxdim tensor, thus the need for triple indexing
        q_list = [q_board[0][x][y] for (x,y) in new]
        # print(q_list)
        
        # Assume an epsilon greedy policy for action selection from avail locations
        if random.random() >= epsilon:
            next_action_loc = new[np.argmax(q_list)]
        else:
            next_action_loc = random.choice(new)
        # print(next_action_loc)
        
    return next_action_loc, action_is_flag

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

def update_network_from_simple_buffer(input_variables, q_network, num_episodes_per_update, fits_per_buffer_fill):
    """
    Run through a single game starting with a new board.
    
    Choose batch_fraction of the state transitions to use to update the q_network
    
    General Flow of state transitions and network updates:
    1)  Network.predict(State 1) -> q1
    2)  get_greedy_next_action(q1, state1) gives next action from unclicked points in state 1
    3)  State1 with next action is now State 2
    4)  Network.predict(State 2) -> q2
    5)  q_val_upate(q1, q2) outputs q1_update
    6)  put (state1, q1_update) into episode buffer
    7)  state 1 <- state 2
    8)  Return to step 1 until end of episode (all spaces clicked)
    9)  Full episode of (state,q_updated) pairs is randomly partitioned, and a batch of pairs is used to update the network.
    10) Generate new board and return to step 1
    
    """
    dimension, num_mines, learning_param, batch_fraction, epsilon = input_variables
    state_counter = 1
    avg_game_time = 0
    history = {}
    history_terminals = {}
    loss = []
    accuracy = []
    
    for game_num in range(num_episodes_per_update):
        if game_num in [num_episodes_per_update//4 -1, 2*num_episodes_per_update//4 -1 , 
                        3*num_episodes_per_update//4-1, num_episodes_per_update -1]:
            print(f"Starting Training Game {game_num + 1} out of {num_episodes_per_update}")
        game_over = False
        start = time.time()
        ref_board = ms.make_board(dimension, num_mines)
        reward_board = ms.make_reward_board(ref_board)
        state_t1 = np.zeros((dimension, dimension, 11))
        
        while not game_over:
            state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
            state_t1_q_board = q_network.predict(state_t1_tensor)
            next_action_loc, _ = get_greedy_next_action(state_t1_q_board, state_t1, epsilon)
            
            state_t2 = one_hot_encode_next_state(state_t1, ref_board, next_action_loc)
            state_t2_tensor = tf.convert_to_tensor(np.expand_dims(state_t2, axis=0))
            state_t2_q_board = q_network.predict(state_t2_tensor)
            
            #Specify that boards with everything clicked need to have a total Q val coverage of 0.
            # This sets the q values when the end board is the next state, being used as a maxq updater
            if (state_counter % (dimension**2-2)) == 0:
                state_t2_q_board = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
            state_t1_q_update = q_value_update(state_t1_q_board, state_t2_q_board, reward_board, learning_param)
            history[state_counter] = (state_t1, state_t1_q_update)
            
            # This is for updating the state/boardpair list when the state is the terminal state
            if (state_counter % (dimension**2-1)) == 0:
                state_t1_q_update = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
                history_terminals[state_counter] = (state_t1, state_t1_q_update)
                game_over = True
                
            history[state_counter] = (state_t1, state_t1_q_update)
            state_counter += 1
            state_t1 = state_t2
            
        end = time.time()
        new_game_time = end-start
        avg_game_time = ms.avg_time_per_game(avg_game_time, new_game_time, game_num+1)

    # Update network with new random subbatches from current buffer
    for _ in range(fits_per_buffer_fill):
        regular_batch = random.sample(history.items(), 4*(dimension ** 2))
        terminal_batch = random.sample(history_terminals.items(), 4)
        _, (terminal, terminal_q) = terminal_batch[0]
        states = [terminal]
        labels = [terminal_q]
        for k, (s,l) in regular_batch:
            states.append(s)
            labels.append(l)
        states = tf.convert_to_tensor(states)
        labels = tf.convert_to_tensor(labels)
        
        # Force model.fit function to use the whole batch at once for a single pass
        # Is it jank? Yes, but IDK how to use the reg and term lists in the built in batch options
        baby_batch_length = dimension ** 2 + 1
        
        # ************
        # How does this work for updating the values inside the q_networks passed in lists?
        outputs = q_network.fit(states, labels, batch_size = baby_batch_length)
        loss.extend(outputs.history['loss'])
        accuracy.extend(outputs.history['accuracy'])
        # *******
    metrics = [loss, accuracy]
    return q_network, avg_game_time, metrics


def update_network_with_tiered_buffer(input_variables, buffer_variables, q_network):

    """
    Currently only working with single q network for simplicity
    """
    
    dimension, num_mines, learning_param, _ , epsilon = input_variables
    generations_per_buffer, games_per_buffer, fits_per_buffer_fill, num_buffer_refills = buffer_variables
    state_counter = 1
    avg_game_time = 0
    history = {}
    history_terminals = {}
    
    """
    Buffer state count = X games => X * dim**2 states stored
    For buffer to contain 3 generations, a single filling of the buffer would need 2/5ths of buffer state length
    State counter continually increases
    
    Use small nums for simple mathz
        Buffer state count = 10
        Fill with 3 generations means num_states_per_update = [2 / (2*gen - 1)] * buffer_state_count
        ie 0-3 is gen1
        4-7 is gen 2
        8,9 is gen 3
            must have states 10,11 be turned into 0,1
            so it's state_counter % buffer_state_count as the key for ID in history
    """
    # Replay Buffer Specifics
    max_buffer_state_count = (dimension ** 2) * games_per_buffer
    num_games_per_update = int((2/(2*generations_per_buffer - 1)) * games_per_buffer)
    num_buffer_updates = num_buffer_refills * generations_per_buffer
            
    for buffer_round in range(num_buffer_updates):
        print(f"==> Starting training on generation {buffer_round + 1} out of {num_buffer_updates}")
        
        # Fill the history buffer
        for game_num in range(num_games_per_update):
            if game_num in [num_games_per_update//4 -1, 2*num_games_per_update//4 -1 , 
                                    3*num_games_per_update//4-1, num_games_per_update -1]:
                print(f"Starting Training Game {game_num + 1} out of {num_games_per_update}")
                print(f"Length of Buffer: {len(history)}")
            start = time.time()
            ref_board = ms.make_board(dimension, num_mines)
            reward_board = ms.make_reward_board(ref_board)
            game_over = False
            state_count_start = state_counter
            state_t1 = np.zeros((dimension, dimension, 11))
            
            while not game_over and state_counter < (state_count_start + dimension ** 2):
                
                # Special ID update for refilling an overflowing buffer
                history_id = state_counter%max_buffer_state_count
                # if (state_counter % (dimension**2-2)) == 0:
                #     print(state_counter, history_id)
                #     print(len(history_terminals))
                
                state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
                state_t1_q_board = q_network.predict(state_t1_tensor)
                next_action_loc, _ = get_greedy_next_action(state_t1_q_board, ref_board, epsilon)
                    
                state_t2 = one_hot_encode_next_state(state_t1, ref_board, next_action_loc)
                state_t2_tensor = tf.convert_to_tensor(np.expand_dims(state_t2, axis=0))
                state_t2_q_board = q_network.predict(state_t2_tensor)
                
                #Specify that boards with everything clicked need to have a total Q val coverage of 0.
                # This sets the q values when the end board is the next state, being used as a maxq updater
                if (state_counter % (dimension**2-2)) == 0:
                    state_t2_q_board = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
                state_t1_q_update = q_value_update(state_t1_q_board, state_t2_q_board, reward_board, learning_param)
                history[history_id] = (state_t1, state_t1_q_update)
                
                # This is for updating the state/boardpair list when the state is the terminal state
                if (state_counter % (dimension**2-1)) == 0:
                    state_t1_q_update = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
                    history_terminals[state_counter] = (state_t1, state_t1_q_update)
                    game_over = True
                    
                state_counter += 1
                state_t1 = state_t2
            end = time.time()
            new_game_time = end-start
            avg_game_time = ms.avg_time_per_game(avg_game_time, new_game_time, game_num+1)

        # Update network with new random subbatches from current buffer
        for _ in range(fits_per_buffer_fill):
            regular_batch = random.sample(history.items(), 4* (dimension ** 2))
            terminal_batch = random.sample(history_terminals.items(), 4)
            _, (terminal, terminal_q) = terminal_batch[0]
            states = [terminal]
            labels = [terminal_q]
            for k, (s,l) in regular_batch:
                states.append(s)
                labels.append(l)
            states = tf.convert_to_tensor(states)
            labels = tf.convert_to_tensor(labels)
            
            # Force model.fit function to use the whole batch at once for a single pass
            # Is it jank? Yes, but IDK how to use the reg and term lists in the built in batch options
            baby_batch_length = dimension ** 2 + 1
            
            # ************
            # How does this work for updating the values inside the q_networks passed in lists?
            q_network.fit(states, labels, batch_size = baby_batch_length)
            # *******

    # Does returning the reference to the input list also return the updated network values?
    return q_network, avg_game_time

def train_q_network_v1(input_variables, num_training_times, q_network, num_episodes_per_update, fits_per_buffer_fill):
    avg_game_times = 0
    metrics = [[], []]
    for i in range(num_training_times):
        print(f"\n==> Single Q v1:\n\tTraining Round #{i+1} out of {num_training_times} <==")
        q_network, avg_game_time, metric = update_network_from_simple_buffer(input_variables, q_network, num_episodes_per_update, fits_per_buffer_fill)
        metrics[0].extend(metric[0])
        metrics[1].extend(metric[1])
        avg_game_times = ms.avg_time_per_game(avg_game_times, avg_game_time, i+1)
    return q_network, avg_game_times, metrics

def train_q_network_v2(input_variables, buffer_variables, num_training_times, q_network):
    avg_game_times = 0
    for i in range(num_training_times):
        print(f"\n==> Single Q v2:\n\tTraining Round #{i+1} out of {num_training_times} <==")
        q_network, avg_game_time = update_network_with_tiered_buffer(input_variables, buffer_variables, q_network)
        avg_game_times = ms.avg_time_per_game(avg_game_times, avg_game_time, i+1)
    return q_network, avg_game_times


def play_one_game_random_choice_baseline(dimension, num_mines):
    
    ref = ms.make_board(dimension, num_mines)
    state_counter = 0
    mine_times = []
    game_over = False 
    state = np.zeros((dimension, dimension, 11))

    while not game_over and state_counter < dimension**2:
        
        # Find all available locations
        locs = np.where(ref == 0)
        places = list(zip(locs[0], locs[1], locs[2]))
        new = [(x,y) for (x,y,z) in places if z == 0]
        if len(new) == 0:
            game_over = True
            
        # Random Choose from list
        next_action_loc = random.choice(new)
        
        if not game_over:
            x,y = next_action_loc
            if ref[x][y][1] == 1:
                mine_times.append(state_counter)
            state = one_hot_encode_next_state(state, ref, next_action_loc, playing = True)
            state_counter += 1    
            
    # This helps score the no flag, continued play version
    avg_mine_click = np.mean(mine_times)
    return avg_mine_click

def play_one_game_single_network(dimension, num_mines, q_network):
    
    ref = ms.make_board(dimension, num_mines)
    state_counter = 0
    mine_times = []
    game_over = False 
    state = np.zeros((dimension, dimension, 11))

    while not game_over and state_counter < dimension**2:
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        state_q_board = q_network.predict(state_tensor)
        next_action_loc, _ = get_greedy_next_action(state_q_board, ref, epsilon = 0)
        if state_counter < dimension ** 2:
            x,y = next_action_loc
            if ref[x][y][1] == 1:
                mine_times.append(state_counter)
            state = one_hot_encode_next_state(state, ref, next_action_loc, playing = True)
            state_counter += 1    
            
    # This helps score the no flag, continued play version
    avg_mine_click = np.mean(mine_times)
    return avg_mine_click


if __name__ == '__main__':
    
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
    # num_training_times_v1 = 1
    # batch_fraction = 1          #Divide the states by this value to produce the batch
    
    # # Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
    # games_per_buffer = 15
    # generations_per_buffer = 2
    # fits_per_generation = 1
    # num_buffer_refills = 1
    # num_training_times_q = 1
    
    # # Evaluation Specifics
    # num_games = 200


    # -----> Actual Data Collection Values <-----  #
    # Training Specifics (non fancy buffer) (only used for one version of regular q)
    num_episodes_per_update = 40
    num_training_times_v1 = 8
    batch_fraction = 1          #Divide the states by this value to produce the batch
    
    # Training Specifics (fancy buffer!) (using small batch sizes, multiple trained predicting generations per buffer)
    games_per_buffer = 100
    generations_per_buffer = 2
    fits_per_generation = 20
    num_buffer_refills = 3
    num_training_times_q = 1
        
    # Evaluation Specifics
    num_games = 400
    
    ##############################################################
    
    
    num_mines = int(mine_percent * (dimension**2))
    input_variables = (dimension, num_mines, learning_param, batch_fraction, epsilon_greedy)
    buffer_variables = (generations_per_buffer, games_per_buffer, fits_per_generation, num_buffer_refills)
    
    input_shape = (dimension, dimension, 11)
    
    
    # Temp Network Setup
    q_network_v1 = tf.keras.Sequential([
                    tf.keras.Input(shape = input_shape),
                    layers.Flatten(),
                    layers.Dropout(0.25),
                    layers.Dense(100*(dimension**2)),
                    layers.Dropout(0.25),
                    layers.Dense(dimension**2),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
    q_network_v1.summary()
    
    q_network_v1.compile(optimizer='adam',
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=['accuracy'])
    
    q_network_v1, q_v1_avg_game_time, _ = train_q_network_v1(input_variables, num_training_times_v1, q_network_v1, 
                                                              num_episodes_per_update, fits_per_generation)
    # # Temp Network Setup
    # q_network_v2 = tf.keras.Sequential([
    #                 tf.keras.Input(shape = input_shape),
    #                 layers.Flatten(),
    #                 layers.Dropout(0.25),
    #                 layers.Dense(100*(dimension**2)),
    #                 layers.Dropout(0.25),
    #                 layers.Dense(dimension**2),
    #                 layers.Reshape(target_shape=(dimension, dimension))
    # ])
    # # q_network_v2.summary()
    
    # q_network_v2.compile(optimizer='adam',
    #                   loss=tf.keras.losses.MeanSquaredError(),
    #                   metrics=['accuracy'])
    
    # q_network_v2, q_v2_avg_game_time = train_q_network_v2(input_variables, buffer_variables,
    #                                                             num_training_times_q, q_network_v2)
    
    avg_clicks_random = []
    avg_clicks_buffer_v1 = []
    # avg_clicks_buffer_v2 = []
    
    vals = [x*num_games//10 for x in range (11)]
    for x in range(0,num_games):
        if x in vals:
            print(f"Starting Game: {x+1} out of {num_games}")
        avg_click_random = play_one_game_random_choice_baseline(dimension, num_mines)
        avg_click_buffer_v1 = play_one_game_single_network(dimension, num_mines, q_network_v1)
        # avg_click_buffer_v2 = play_one_game_single_network(dimension, num_mines, q_network_v2)
        
        # Data Aggregation
        avg_clicks_random.append(avg_click_random)
        avg_clicks_buffer_v1.append(avg_click_buffer_v1)
        # avg_clicks_buffer_v2.append(avg_click_buffer_v2)
        
    avg_score_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]
    avg_score_buffer_v1 = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_buffer_v1]
    # avg_score_buffer_v2 = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_buffer_v2]
    
    display_data = []
    display_data.append(avg_score_random)
    display_data.append(avg_score_buffer_v1)
    # display_data.append(avg_score_buffer_v2)
    mean_scores = [round(np.mean(score), 1) for score in display_data]
    
    count = 0
    for dat in display_data:
        for val in dat:
            if val < 0 or val > 100:
                print(val)
                count += 1
    print("Total Out of Bounds:", count)
    
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
        

    a_out, b_out = distinct_vals_b_minus_a(avg_score_random, avg_score_buffer_v1)      
    
    
    plt.bar(a_out.keys(), a_out.values())
    plt.bar(b_out.keys(), b_out.values())
    plt.legend(labels = ['Random Choice', 'Q Buffer v1'])
    plt.show()
    
    # Plot the things!
    sns.violinplot(data = display_data, inner = 'quartile')
    # plt.hist(display_data, color=['r','b', 'g'], alpha=0.5)
    plt.title(f"\nTraining -> Learning Rate: {learning_param} || Epsilon Greedy: {epsilon_greedy}" + 
              f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
              "\n"+ "- "*20 + 
              f"\nRandom choice and trained average scores over {num_games} games" +
              f"\nRandom Choice: {mean_scores[0]}/100 || Median: {round(np.median(avg_score_random), 1)} || StDev: {round(np.std(avg_score_random), 1)}" +
              f"\nBuffered Q v1: {mean_scores[1]}/100 || Median: {round(np.median(avg_score_buffer_v1), 1)} || StDev: {round(np.std(avg_score_buffer_v1), 1)}"
               # + f"\nBuffered Q v2: {mean_scores[2]}/100 || Median: {round(np.median(avg_score_buffer_v2), 1)} || StDev: {round(np.std(avg_score_buffer_v2), 1)}" 
              )
              # + f"\nQv1 Avg Time: {round(q_v1_avg_game_time, 2)} sec(s) || Qv2 Avg Time: {round(q_v2_avg_game_time, 2)} sec(s)")
    
    # plt.legend(labels = ['Random Choice', 'Q Buffer v1'])
    plt.xticks(ticks = [0,1], labels = ['Random Choice', 'Q Buffer v1'])
    plt.ylabel('Game Score [Percent of Optimal Moves Made]')
    plt.show()