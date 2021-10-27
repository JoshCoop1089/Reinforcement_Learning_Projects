# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 18:59:56 2021

@author: joshc
"""
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
import copy

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

def create_q_network(dimension, num_mines, dense_size, dropout_coef):
    input_shape = (dimension, dimension, 11)
    
    q_network = tf.keras.Sequential([
                tf.keras.Input(shape = input_shape),
                layers.Flatten(),
                layers.Dropout(dropout_coef),
                layers.Dense(dense_size),
                layers.Dropout(dropout_coef),
                layers.Dense(dimension**2, activation = 'relu'),
                layers.Reshape(target_shape=(dimension, dimension))
    ])
    
    q_network.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    return q_network

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
    next_action_loc = (0,0)
        
    # Get list of all available spots
    locs = np.where(board[:,:,0] == 0)
    places = list(zip(locs[0], locs[1]))
    if len(places) == 0:
        pass
        
    # Get location of highest q in available locations
    else:
        # q_board is a 1xdimxdim tensor, thus the need for triple indexing
        q_list = [q_board[0][x][y] for (x,y) in places]
        
        # Assume an epsilon greedy policy for action selection from avail locations
        if random.random() < epsilon:
            next_action_loc = places[np.argmax(q_list)]
        else:
            next_action_loc = random.choice(places)
        
    return next_action_loc

def generate_q_test_data(q_training_parameters, epsilon, q_network):
    dimension, num_mines, num_games_per_training, learning_param = q_training_parameters
    state_counter = 1
    history = {}
    history_terminals = {}
    
    for game_num in range(num_games_per_training):
        if game_num in [num_games_per_training//4 -1, 2*num_games_per_training//4 -1 , 
                        3*num_games_per_training//4-1, num_games_per_training -1]:
            print(f"Starting Training Game {game_num + 1} out of {num_games_per_training}")
        game_over = False
        ref_board = ms.make_board(dimension, num_mines)
        reward_board = ms.make_reward_board(ref_board)
        state_t1 = np.zeros((dimension, dimension, 11))
        
        while not game_over:
            state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
            state_t1_q_board = q_network.predict(state_t1_tensor)
            next_action_loc = get_greedy_next_action(state_t1_q_board, state_t1, epsilon)
            
            state_t2 = one_hot_encode_next_state(state_t1, ref_board, next_action_loc)
            state_t2_tensor = tf.convert_to_tensor(np.expand_dims(state_t2, axis=0))
            state_t2_q_board = q_network.predict(state_t2_tensor)
            
            #Specify that boards with everything clicked need to have a total Q val coverage of 0.
            # This sets the q values when the end board is the next state, being used as a maxq updater
            if (state_counter % (dimension**2)) == (dimension**2-1):
                state_t2_q_board = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
              
            # This is for updating the state/boardpair list when the state is the terminal state
            if (state_counter % (dimension**2)) == 0:
                state_t1_q_update = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
                history_terminals[state_counter] = (state_t1, state_t1_q_update)
                game_over = True
                
            # Otherwise, it's just a normal non terminal board
            else:
                state_t1_q_update = q_value_update(state_t1_q_board, state_t2_q_board, reward_board, learning_param)
                history[state_counter] = (state_t1, state_t1_q_update)
                
            state_counter += 1
            state_t1 = state_t2
    
    return history, history_terminals

def train_actor_critic_networks(q_training_parameters, actor_network, critic_network):

    # checkpoint_filepath = '\model_checkpoints\checkpoint_best'
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #                                 filepath=checkpoint_filepath,
    #                                 save_weights_only=True,
    #                                 monitor='accuracy',
    #                                 mode='max',
    #                                 save_best_only=True)

    full_loss = []
    full_accuracy = []
    dimension, num_mines, num_games_per_training, learning_param = q_training_parameters
    for epsilon_counter in range(0,21):
        epsilon = epsilon_counter * 5/100
        loss_plateau_reached = False
        print("---> Chance of using actor_network: ", epsilon, " <---")
        while not loss_plateau_reached:
            plateau_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta = 2e-4, patience=3)
            
            # Generate training data
            history, history_terminals = generate_q_test_data(q_training_parameters, epsilon, actor_network)
            
            # Sample training data to fit to network
            state = []
            label = []
            sample_states =  int(num_games_per_training * 0.02) + 1
            terminal_batch = random.sample(history_terminals.items(), sample_states)
            for k, (s,l) in terminal_batch:
                state.append(s)
                label.append(l)
            batch = random.sample(history.items(), sample_states*(dimension**2))
            for k, (s,l) in batch:
                state.append(s)
                label.append(l)
            states = tf.convert_to_tensor(state)
            labels = tf.convert_to_tensor(label)
            
            output = critic_network.fit(states, labels, epochs = 5, callbacks = [plateau_callback])
            full_loss.extend(output.history["loss"])
            full_accuracy.extend(output.history['loss'])
            
            # If loss length is less than the number of epochs, a plateau was reached
            # Stop training, go back to for loop and increase epsilon_chance
            temp_loss = output.history["loss"]
            if len(temp_loss) < 5:
                loss_plateau_reached = True
    
    metrics = [full_loss, full_accuracy]
    return actor_network, critic_network, metrics


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
        next_action_loc = get_greedy_next_action(state_q_board, ref, epsilon = 1)
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
    # Board Specifics
    dimension = 5    
    mine_percent = 0.4
    
    # ############################
    # ## Testing Output Numbers ##

    # # Q Network Specifics 
    # dense_size = 80*(dimension**2)
    # dropout_coef = 0.375
    # learning_param = 0.05
    # num_games_per_q_training = 20
    # actor_critic_training_times = 2
    
    # # Evaluation
    # num_games = 20
    # #############################
    

    #############################
    ## Actual Data Run Numbers ##
    
    # Q Network Specifics 
    dense_size = 80*(dimension**2)
    dropout_coef = 0.375
    learning_param = 0.05
    num_games_per_q_training = 100
    actor_critic_training_times = 10
    
    # Evaluation
    num_games = 1000
    #############################
    
    num_mines = int(mine_percent * (dimension**2))
    q_training_parameters = dimension, num_mines, num_games_per_q_training, learning_param
    
    # Create base networks, and pretrain mine network
    actor_network = create_q_network(dimension, num_mines, dense_size, dropout_coef)   
    critic_network = create_q_network(dimension, num_mines, dense_size, dropout_coef)   
    
    training_loss = []
    training_accuracy = []
    for training_round in range(actor_critic_training_times):
        print("---> Starting AC Training Round", training_round, "<---")
        actor_network, critic_network, [loss, accuracy] = \
                        train_actor_critic_networks(q_training_parameters, actor_network, critic_network)
        training_loss.extend(loss)
        training_accuracy.extend(accuracy)
        actor_network.set_weights(critic_network.get_weights())
    
    # Play series of games to judge competence compared to random
    avg_clicks_random = []
    avg_clicks_q_network = []
    
    vals = [x*num_games//10 for x in range (11)]
    for x in range(1,num_games+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games}")
        avg_click_random = play_one_game_random_choice_baseline(dimension, num_mines)
        avg_click_q_network = play_one_game_single_network(dimension, num_mines, actor_network)
        
        # Data Aggregation
        avg_clicks_random.append(avg_click_random)
        avg_clicks_q_network.append(avg_click_q_network)
        
    avg_score_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]
    avg_score_q_network = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_q_network]
    
    display_data = []
    display_data.append(avg_score_random)
    display_data.append(avg_score_q_network)
    mean_scores = [round(np.mean(score), 1) for score in display_data]
       
    # Plot the things!
    sns.violinplot(data = display_data, inner = 'quartile')
    plt.title(f"\nTraining -> Learning Rate: {learning_param}" + 
              f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
              "\n"+ "- "*20 + 
              f"\nRandom choice and trained average scores over {num_games} games" +
              f"\nRandom Choice: {mean_scores[0]}/100 || Median: {round(np.median(avg_score_random), 1)} || StDev: {round(np.std(avg_score_random), 1)}" +
              f"\nActorCritic: {mean_scores[1]}/100 || Median: {round(np.median(avg_score_q_network), 1)} || StDev: {round(np.std(avg_score_q_network), 1)}" 
              )
    
    plt.xticks(ticks = [0,1], labels = ['Random Choice', 'Actor-Critic'])
    plt.ylabel('Game Score [Percent of Optimal Moves Made]')
    plt.show()
    
    # Compare Random to Q_Learning
    a_out, b_out = distinct_vals_b_minus_a(avg_score_random, avg_score_q_network)         
    plt.bar(a_out.keys(), a_out.values())
    plt.bar(b_out.keys(), b_out.values())
    plt.legend(labels = ['Random Choice', 'Actor_Critic'])
    plt.show()
        
    # Training Loss Over Time!
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss, 'bo')
    plt.show()