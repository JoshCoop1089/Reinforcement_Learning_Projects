# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 13:10:00 2021

@author: joshc
"""
import tensorflow as tf
import numpy as np
import MineSweeper_Base_Game as ms
import seaborn as sns
import matplotlib.pyplot as plt
import random


from Use_Mine_Predictions_As_Training_Limited_Epsilon import one_hot_encode_next_state
from Use_Mine_Predictions_As_Training_Limited_Epsilon import get_greedy_next_action
from Use_Mine_Predictions_As_Training_Limited_Epsilon import create_and_train_mine_predictor
from Use_Mine_Predictions_As_Training_Limited_Epsilon import mine_prediction_network_next_action
from Use_Mine_Predictions_As_Training_Limited_Epsilon import q_value_update
from Use_Mine_Predictions_As_Training_Limited_Epsilon import create_q_network

def make_reward_board_lose_on_first_mine(board):
    reward_board = np.ones(board.shape[0:2])
    dimension = board.shape[0]
    # Reward for mine is -1, otherwise 1
    # Mine's are encoded as a 1 on board, 0 for clear, so  1 - (2 * val) produces proper reward layout
    for x in range(dimension):
        for y in range(dimension):
            reward_board[x][y] -= 2 * board[x][y][1]
    return reward_board


def generate_q_test_data(q_training_parameters, mine_network_percent, q_network, mine_prediction_model):
    state_counter = 1
    history = {}
    history_terminals = {}
    dimension, num_mines, num_games_per_training, learning_param, min_delta = q_training_parameters
    
    vals = [x*num_games_per_training//10 for x in range (11)]
    for game_num in range(1,num_games_per_training+1):
        if game_num in vals:
            print(f"Starting Training Game {game_num + 1} out of {num_games_per_training}")
        game_over = False
        ref_board = ms.make_board(dimension, num_mines)
        reward_board = make_reward_board_lose_on_first_mine(ref_board)
        state_t1 = np.zeros((dimension, dimension, 11))
        
        # Early quit check for mine location on first turn
        state_t1_temp = np.zeros((dimension, dimension, 11))
        state_t1_temp_tensor = tf.convert_to_tensor(np.expand_dims(state_t1_temp, axis=0))
        state_t1_q_temp = q_network.predict(state_t1_temp_tensor)
        
        while not game_over:
            state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
            state_t1_q_board = q_network.predict(state_t1_tensor)

            # Use the trained mine prediction network to get the next location
            next_action_loc = mine_prediction_network_next_action(state_t1_tensor, ref_board, mine_network_percent, mine_prediction_model)
            
            # Check is location is mined
            x,y = next_action_loc
            if ref_board[x][y][1] == 1:
                state_t1_q_update = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
                history_terminals[state_counter] = (state_t1, state_t1_q_update)
                
                # Update previous state as well to account for new terminal
                state_t0_q_update = q_value_update(state_t1_q_temp, state_t1_q_board, reward_board, learning_param)
                history[state_counter-1] = (state_t1_temp, state_t0_q_update)
                game_over = True
                break
            
            else:
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
            state_t1_temp = state_t1
            state_t1_q_temp = state_t1_q_board
            state_t1 = state_t2
    
    return history, history_terminals

def train_q_network(q_training_parameters, q_network, mine_prediction_model):

    full_loss = []
    full_accuracy = []
    dimension, num_mines, num_games_per_training, learning_param, min_delta = q_training_parameters
    for use_mine_network_counter in range(15,21):
        mine_network_percent = use_mine_network_counter * 5/100
        loss_plateau_reached = False
        print("---> Chance of using mine_predictor: ", mine_network_percent, " <---")
        
        # Generate training data
        fit_counter = 0
        history, history_terminals = generate_q_test_data(q_training_parameters, mine_network_percent, q_network, mine_prediction_model)
        
        while not loss_plateau_reached:
            checkpoint_filepath = '\model_checkpoints\checkpoint_best'
            best_model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=checkpoint_filepath,
                                            save_weights_only=True,
                                            monitor='accuracy',
                                            mode='max',
                                            save_best_only=True)
            
            plateau_callback = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta = min_delta, patience=3)
            
            # Sample training data to fit to network
            state = []
            label = []
            sample_states =  int(0.05*len(history_terminals))+1
            print(f"Attempting to sample {sample_states*(dimension**2)} from history buffer of length {len(history)}")
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
            
            
            output = q_network.fit(states, labels, callbacks = [best_model_checkpoint_callback])
            full_loss.extend(output.history["loss"])
            full_accuracy.extend(output.history['accuracy'])
            fit_counter += 1
            
            # Allow for 20 fittings with new epoch level, then check for plateau every new fitting
            if fit_counter > 20:
                q_network.load_weights(checkpoint_filepath)

                output = q_network.fit(states, labels, epochs = 6, callbacks = [plateau_callback])
                full_loss.extend(output.history["loss"])
                full_accuracy.extend(output.history['accuracy'])
                
                # If loss length is less than the number of training rounds, a plateau was reached
                # Stop training, go back to for loop and increase mine_network_percent_chance
                temp_loss = output.history["loss"]
                if len(temp_loss) < 6:
                    loss_plateau_reached = True
    
    metrics = [full_loss, full_accuracy]
    return q_network, metrics

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

def play_one_game_random_choice_baseline_lose_on_first_mine(dimension, num_mines):
    
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
            
            # Hit a mine, instant loss, record score
            if ref[x][y][1] == 1:
                mine_times = state_counter
                return mine_times
               
            # Otherwise, keep playing
            state = one_hot_encode_next_state(state, ref, next_action_loc, playing = True)
            state_counter += 1    
        
    # If you somehow make it to the end, score is state_counter
    return state_counter

def play_one_game_single_network_lose_on_first_mine(dimension, num_mines, q_network):
    
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
            
            # Hit a mine, instant loss, record score
            if ref[x][y][1] == 1:
                mine_times = state_counter 
                return mine_times
            
            # Otherwise, keep playing
            state = one_hot_encode_next_state(state, ref, next_action_loc, playing = True)
            state_counter += 1    
            
    # If you somehow make it to the end, score is state_counter
    return state_counter

def play_one_game_mine_predict_network_lose_on_first_mine(dimension, num_mines, mine_prediction_model):
    
    board = ms.make_board(dimension, num_mines)
    b_enc = np.zeros((dimension, dimension, 11))
    count = 0
    mine_times = []
    
    while count < dimension ** 2:
        b_enc_t = tf.convert_to_tensor(np.expand_dims(b_enc, axis=0))
        pred = mine_prediction_model.predict(b_enc_t)
        locs = np.where(board[:,:,0] == 0)
        places = list(zip(locs[0], locs[1]))
        actions = [pred[0][x][y] for (x,y) in places]
        
        if count < dimension ** 2:
            next_loc = places[np.argmin(actions)]
            if board[next_loc[0]][next_loc[1]][1] == 1:
                mine_times = count
                return mine_times
            b_enc = one_hot_encode_next_state(b_enc, board, next_loc)
            count += 1
    
    return count

def mine_predictor_fed_q_learning_limited_epsilon_runs_lose_on_first_mine(q_network_params, q_training_params, mine_training_params, num_games_eval):
    dimension, mine_percent, dense_size, dropout_coef = q_network_params
    learning_param, num_games_per_epsilon_level_training, num_games_per_epsilon_level_training, min_delta = q_training_params
    num_games_mine_train, num_training_rounds, mine_dropout_coef, mine_dense_size = mine_training_params
    
    num_mines = int(mine_percent * (dimension**2))
    mine_network_training_params = [num_games_mine_train, num_training_rounds, mine_dropout_coef, mine_dense_size]
    q_training_parameters = dimension, num_mines, num_games_per_epsilon_level_training, learning_param, min_delta
    
    # Create base networks, and pretrain mine network
    q_network = create_q_network(dimension, num_mines, dense_size, dropout_coef)   
    mine_prediction_model = create_and_train_mine_predictor(dimension, num_mines, mine_network_training_params)
    
    
    # To train the q network, we use the board predictions from mine network to choose 
        #the next location, and depending on the epsilon value, either take that move 
        #or a random one
        
    q_network, [training_loss, training_accuracy] = train_q_network(q_training_parameters, q_network, mine_prediction_model)
    
    # Use networks to play series of games to judge competence compared to random
    avg_clicks_random = []
    avg_clicks_mine_predict = []
    avg_clicks_q_network = []
    
    vals = [x*num_games_eval//10 for x in range (11)]
    for x in range(1,num_games_eval+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games_eval}")
        avg_click_random = play_one_game_random_choice_baseline_lose_on_first_mine(dimension, num_mines)
        avg_click_mine_predict = play_one_game_mine_predict_network_lose_on_first_mine(dimension, num_mines, mine_prediction_model)
        avg_click_q_network = play_one_game_single_network_lose_on_first_mine(dimension, num_mines, q_network)
        
        # Data Aggregation
        avg_clicks_random.append(avg_click_random)
        avg_clicks_mine_predict.append(avg_click_mine_predict)
        avg_clicks_q_network.append(avg_click_q_network)
        
    avg_score_random = [100*score/(dimension**2) for score in avg_clicks_random]
    avg_score_mine_predict = [100*score/(dimension**2) for score in avg_clicks_mine_predict]
    avg_score_q_network = [100*score/(dimension**2) for score in avg_clicks_q_network]
    
    display_data = []
    display_data.append(avg_score_random)
    display_data.append(avg_score_mine_predict)
    display_data.append(avg_score_q_network)
    mean_scores = [round(np.mean(score), 1) for score in display_data]
       
    # Plot the things!
    sns.violinplot(data = display_data, inner = 'quartile')
    # plt.hist(display_data, color=['r','b', 'g'], alpha=0.5)
    plt.title(f"\nTraining -> Learning Rate: {learning_param} || Min_Delta for Loss Plateau: {min_delta}" + 
              f"\n Board Size: {dimension}x{dimension} || Mines: {num_mines}"
              "\n"+ "- "*20 + 
              f"\nRandom choice and trained average scores over {num_games_eval} games" +
              f"\nRandom Choice: {round(np.mean(avg_clicks_random), 1)}/{dimension**2} {mean_scores[0]}% || Median: {round(np.median(avg_score_random), 1)} || StDev: {round(np.std(avg_score_random), 1)}" 
              + f"\nMine Predictor: {round(np.mean(avg_clicks_mine_predict), 1)}/{dimension**2}, {mean_scores[1]}%|| Median: {round(np.median(avg_score_mine_predict), 1)} || StDev: {round(np.std(avg_score_mine_predict), 1)}"
              + f"\nQ Learning: {round(np.mean(avg_clicks_q_network), 1)}/{dimension**2}, {mean_scores[2]}% || Median: {round(np.median(avg_score_q_network), 1)} || StDev: {round(np.std(avg_score_q_network), 1)}" 
              )
    
    # plt.legend(labels = ['Random Choice', 'Q Buffer v1'])
    plt.xticks(ticks = [0,1,2], labels = ['Random Choice', 'Mine Predictor', 'Q Learning'])
    plt.ylabel('Game Score [How many turns did agent survive?]')
    plt.show()
    
    # Compare Random to Mine Predictor
    a_out, b_out = distinct_vals_b_minus_a(avg_score_random, avg_score_mine_predict)         
    plt.bar(a_out.keys(), a_out.values())
    plt.bar(b_out.keys(), b_out.values())
    plt.legend(labels = ['Random Choice', 'Mine_prediction'])
    plt.xlabel('Game Score [How many turns did agent survive?]')
    plt.ylabel('Score Frequency')
    plt.show()
    
    # Compare Random to Q_Learning
    a_out, b_out = distinct_vals_b_minus_a(avg_score_random, avg_score_q_network)         
    plt.bar(a_out.keys(), a_out.values())
    plt.bar(b_out.keys(), b_out.values())
    plt.legend(labels = ['Random Choice', 'Q_network'])
    plt.xlabel('Game Score [How many turns did agent survive?]')
    plt.ylabel('Score Frequency')
    plt.show()
    
    # Compare Mine Predictor to Q-Learning
    a_out, b_out = distinct_vals_b_minus_a(avg_score_q_network, avg_score_mine_predict)         
    plt.bar(a_out.keys(), a_out.values())
    plt.bar(b_out.keys(), b_out.values())
    plt.legend(labels = ['Q_network', 'Mine_prediction'])
    plt.xlabel('Game Score [How many turns did agent survive?]')
    plt.ylabel('Score Frequency')
    plt.show()
    
    # Training Loss Over Time!
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_loss)
    plt.xlabel('Number of Epochs')
    plt.title('Q-Learning Model Loss over Time')
    plt.show()
    
    # Training Accuracy Over Time!
    epochs = range(1, len(training_loss) + 1)
    plt.plot(epochs, training_accuracy)
    plt.xlabel('Number of Epochs')
    plt.title('Q-Learning Model Accuracy over Time')
    plt.show()
