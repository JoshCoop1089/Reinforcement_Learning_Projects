# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:13:38 2021

@author: joshc
"""
import MineSweeper_Base_Game as ms

import random, copy, time
import numpy as np
import tensorflow as tf

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

def get_next_state(board_c, location, flag = False, playing = False):
    # Do we have to deep copy the board state in order for the history states to stay as they were before state transition?
    if not playing:
        board = copy.deepcopy(board_c) 
    else:
        board = board_c

    x,y = location
    if not flag:
        board[x][y][0] = 1
    else:
        board[x][y][0] = -1
    
    return board

def update_network_from_multiple_episodes(input_variables, q_network, num_episodes_per_update):
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
    state_counter = 0
    avg_game_time = 0
    history = {}
    
    for game_num in range(num_episodes_per_update):
        start = time.time()
        state_t1 = ms.make_board(dimension, num_mines)
        reward_board = ms.make_reward_board(state_t1)
        state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
        state_t1_q_board = q_network.predict(state_t1_tensor)
        next_action_loc, _ = get_greedy_next_action(state_t1_q_board, state_t1, epsilon)
        
        while state_counter < dimension**2:
            # print("Going to next state")
            state_t2 = get_next_state(state_t1, next_action_loc, flag = False)
            state_t2_tensor = tf.convert_to_tensor(np.expand_dims(state_t2, axis=0))
            state_t2_q_board = q_network.predict(state_t2_tensor)
            
            #Specify that boards with everything clicked need to have a total Q val coverage of 0.
            # This sets the q values when the end board is the next state, being used as a maxq updater
            if state_counter == dimension**2-2:
                state_t2_q_board = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
            state_t1_q_update = q_value_update(state_t1_q_board, state_t2_q_board, reward_board, learning_param)
            
            # This is for updating the state/boardpair list when the state is the terminal state
            if state_counter == dimension**2-1:
                state_t1_q_update = tf.convert_to_tensor(np.zeros((1,dimension, dimension)))
                
            history[state_counter] = (state_t1, state_t1_q_update)
            state_counter += 1
            state_t1 = state_t2
            state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
            state_t1_q_board = q_network.predict(state_t1_tensor)
            next_action_loc, _ = get_greedy_next_action(state_t1_q_board, state_t1, epsilon)
            
        end = time.time()
        new_game_time = end-start
        avg_game_time = ms.avg_time_per_game(avg_game_time, new_game_time, game_num+1)

    # Select a random number of states from history, and update network
    batch = random.sample(history.items(), (dimension**2)//batch_fraction)
    states = []
    labels = []
    for k, (s,l) in batch:
        states.append(s)
        labels.append(l)
    states = tf.convert_to_tensor(states)
    labels = tf.convert_to_tensor(labels)
    q_network.fit(states, labels, batch_size = dimension**2, epochs = 12)
    return q_network, avg_game_time