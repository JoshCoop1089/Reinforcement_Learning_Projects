# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:51:20 2021

@author: joshc

First Thoughts:

Replay Buffer of Fixed Length
	Use Ordered Dict objects
	Keep structure as counter -> (state, q_board)
	let the buffer be dim**2*(num_episodes/3)
		this allows for the buffer to be filled 3 times per training run
		tweak this length?
	will need to use a state counter % (num_episodes/3) as the key value
	
	When do you use the buffer to select a new batch to train?
		Base state of network before trainign begins is called fit0
		First thought: 
			Let buffer fill 2/3rds using fit0 to predict and update q_board.  
			Pick batch and fit network (calling this fit1 for now)
			Then execute enough games to fill the buffer again by 2/3rds, 
				but this would mean that the first third of the og games is wiped out
			Now the buffer trains on 1/3rd of games from fit 0 and 2/3rds from fit 1 (call this fit 2)
			This cycle repeats, always replacing 2/3rds of the buffer, allowing each new fit to be 
				2/3rd updated network, and 1/3 last gen of network
	
Terminal States vs Regular States
	Do we use terminal state and the terminal-1 state, since technically terminal -1 is being updated from the terminal?

Batch Updates
	How do you select a fixed small number of two different lists and organize them into tensor batch efficiently?
	If tf.fit has a batch size param, would that only work on one single list? 
        (ie only choosing from non terminal state buffer)
	Otherwise, we'd have to do the random selection ourselves and refeed into tf.fit like we currently do.
        Not onerous, but take time to explore tf.fit parameter choices

--------------------------------------------
Second Thoughts:
    
Set buffer size (keep it at 100 games? make it depend on something else?)
figure out how many games per fractional update (pass the fraction in from ms_testing)
identify what network is being used as 
    predictor network
    network being updated
    
for use with doubleQ, do a coin flip at the start of every round

figure out how many batches of games will be needed to fulfill requested game total for this set of training

for round in range(buffer_sets):
    A single round would involve:
        (Double Q) Coin flip to choose network 1 or 2 as "prediction"
        X numbers of games using specific network to create qboards and set updated q_boards
        X number of specific terminal states (one from each game with special all 0 q board)
        
        (Double Q) Need to get prediction from network 1 and network 2 both, and add the boards 
            together to use in epsilon greedy action choice, even if you're only using one of 
            the networks outputs to update q_boards for replay buffer
            
        When adding reg states to buffer, the key ID would be state_counter%buffer_length
        
        When all games finished:
            need to pick random selection of state,q_board pairs from buffer
                set size of random selection as dim**2?
            append 1 randomly chosen terminal state,q_board pair 
        
        Send combined list of states and the single terminal state into network which requires updating
"""
import MineSweeper_Base_Game as ms
# import MineSweeper_TF_Functions_Regular_Q as rq
from MineSweeper_TF_Functions_Regular_Q import get_greedy_next_action, get_next_state, q_value_update

import random, time
import numpy as np
import tensorflow as tf

def double_q_greedy_next_action(state_t1_q_board, state_t1_q_board_2, state_t1, epsilon):
    action_is_flag = False
    next_action_loc = (0,0)
        
    # This is such a janky way of doing this wtf lol
    # Get list of all available spots
    locs = np.where(state_t1 == 0)
    places = list(zip(locs[0], locs[1], locs[2]))
    new = [(x,y) for (x,y,z) in places if z == 0]
    # print(new)
    if len(new) == 0:
        pass
        
    # Get location of highest q in available locations
    else:
        # q_board is a 1xdimxdim tensor, thus the need for triple indexing
        q_list = [state_t1_q_board[0][x][y] + state_t1_q_board_2[0][x][y] for (x,y) in new]
        # print(q_list)
        
        # Assume an epsilon greedy policy for action selection from avail locations
        if random.random() >= epsilon:
            next_action_loc = new[np.argmax(q_list)]
        else:
            next_action_loc = random.choice(new)
        # print(next_action_loc)
        
    return next_action_loc, action_is_flag

def update_network_with_tiered_buffer(input_variables, buffer_variables, q_networks, update_type):
    """
    q_networks would be a list containing either:
        1 network for regular q learning
        2 networks for doubleq and actor critic
        
    update_type is a string indicating which network to choose from q_networks 
        as the predictor and the eventually updated network
        
    function will return the updated regular q network, the updated actor network, or the pair of double q networks
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

    # Regular Q Network
    if update_type == 'RegularQ':
        predict = update = 0
    
    # Actor Critic
    if update_type == 'ActorCritic':
        predict = 0
        update = 1
        
    for buffer_round in range(num_buffer_updates):
        print(f"==> Starting training on generation {buffer_round + 1} out of {num_buffer_updates}")
        
        # Double Q Network (randomly choose which network to update every time to add more to buffer)
        if update_type == "DoubleQ":
            predict = random.randint(0,1)
            update = abs(predict - 1)
        
        # Choosing the correct networks from passed in list based on update type
        predict_q_net = q_networks[predict]
        update_q_net = q_networks[update]
        
        # Fill the history buffer
        for game_num in range(num_games_per_update):
            if game_num in [num_games_per_update//4 -1, 2*num_games_per_update//4 -1 , 
                                    3*num_games_per_update//4-1, num_games_per_update -1]:
                print(f"Starting Training Game {game_num + 1} out of {num_games_per_update}")
                print(f"Length of Buffer: {len(history)}")
            start = time.time()
            state_t1 = ms.make_board(dimension, num_mines)
            reward_board = ms.make_reward_board(state_t1)
            game_over = False
            state_count_start = state_counter
            
            while not game_over and state_counter < (state_count_start + dimension ** 2):
                
                # Special ID update for refilling an overflowing buffer
                history_id = state_counter%max_buffer_state_count
                # if (state_counter % (dimension**2-2)) == 0:
                #     print(state_counter, history_id)
                #     print(len(history_terminals))
                
                state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
                state_t1_q_board = predict_q_net.predict(state_t1_tensor)
                
                # Double Q requires summing up predictions from both networks to choose an action
                if update_type == "DoubleQ":
                    state_t1_q_board_2 = update_q_net.predict(state_t1_tensor)
                    next_action_loc, _ = double_q_greedy_next_action(state_t1_q_board, state_t1_q_board_2,
                                                                                   state_t1, epsilon)
                else:
                    next_action_loc, _ = get_greedy_next_action(state_t1_q_board, state_t1, epsilon)
                    
                state_t2 = get_next_state(state_t1, next_action_loc, flag = False)
                state_t2_tensor = tf.convert_to_tensor(np.expand_dims(state_t2, axis=0))
                state_t2_q_board = predict_q_net.predict(state_t2_tensor)
                
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
            regular_batch = random.sample(history.items(), dimension ** 2)
            terminal_batch = random.sample(history_terminals.items(), 1)
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
            update_q_net.fit(states, labels, batch_size = baby_batch_length)
            # *******

    # Does returning the reference to the input list also return the updated network values?
    return q_networks, avg_game_time