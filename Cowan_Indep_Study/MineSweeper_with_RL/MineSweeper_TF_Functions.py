# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:13:38 2021

@author: joshc
"""
import MineSweeper_Base_Game as ms

import random, copy
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

def get_greedy_next_action(q_board, board):
    """
    The zipping on list_of_locs might need to be changed when flag functionality is added?
    """
    action_is_flag = False
    game_over = False
    next_action_loc = (0,0)
        
    # This is such a janky way of doing this wtf lol
    # Get list of all available spots
    locs = np.where(board == 0)
    places = list(zip(locs[0], locs[1], locs[2]))
    new = [(x,y) for (x,y,z) in places if z == 0]
    if len(new) == 0:
        game_over = True
        
    # Get location of highest q in available locations
    else:
        # q_board is a 1xdimxdim tensor, thus the need for triple indexing
        q_list = [q_board[0][x][y] for (x,y) in new]
        # print(q_list)
        
        # np.argmax only gives the first location in case of a tie in q vals
            # do we need to change this to account for random choice of ties?
        next_action_loc = new[np.argmax(q_list)]
        # print(next_action_loc)
        
    return next_action_loc, action_is_flag, game_over

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
    dimension, num_mines, learning_param, batch_fraction = input_variables
    state_counter = 0
    history = {}
    
    for _ in range(num_episodes_per_update):
        state_t1 = ms.make_board(dimension, num_mines)
        reward_board = ms.make_reward_board(state_t1)
        state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
        state_t1_q_board = q_network.predict(state_t1_tensor)
        next_action_loc, _, game_over = get_greedy_next_action(state_t1_q_board, state_t1)
        
        while not game_over and state_counter < dimension**2:
            # print("Going to next state")
            state_t2 = get_next_state(state_t1, next_action_loc, flag = False)
            state_t2_tensor = tf.convert_to_tensor(np.expand_dims(state_t2, axis=0))
            state_t2_q_board = q_network.predict(state_t2_tensor)
            state_t1_q_update = q_value_update(state_t1_q_board, state_t2_q_board, reward_board, learning_param)
            history[state_counter] = (state_t1, state_t1_q_update)
            state_counter += 1
            state_t1 = state_t2
            state_t1_tensor = tf.convert_to_tensor(np.expand_dims(state_t1, axis=0))
            state_t1_q_board = q_network.predict(state_t1_tensor)
            next_action_loc, _, game_over = get_greedy_next_action(state_t1_q_board, state_t1)

    # Select a random number of states from history, and update network
    batch = random.sample(history.items(), (dimension**2)//batch_fraction)
    states = []
    labels = []
    for k, (s,l) in batch:
        states.append(s)
        labels.append(l)
    states = tf.convert_to_tensor(states)
    labels = tf.convert_to_tensor(labels)
    q_network.fit(states, labels, epochs = 10)
    return q_network

def play_one_game(dimension, num_mines, q_network):
    
    state = ms.make_board(dimension, num_mines)
    state_counter = 0
    mine_times = []
    game_over = False 
        
    while not game_over and state_counter < dimension**2:
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        state_q_board = q_network.predict(state_tensor)
        next_action_loc, _, game_over = get_greedy_next_action(state_q_board, state)
        if not game_over:
            x,y = next_action_loc
            # print("Next Click is: ", next_action_loc)
            if state[x][y][1] == 1:
                mine_times.append(state_counter)
            #     print("Location is a mine")
            # print("Going to next state.")
            # print("- "*dimension)
            state = get_next_state(state, next_action_loc, flag = False, playing = True)
            # ms.print_board(state)
            state_counter += 1    
            
    # This helps score the no flag, continued play version
    avg_mine_click = np.mean(mine_times)
    return avg_mine_click

def play_one_game_random_choice_baseline(dimension, num_mines):
    
    state = ms.make_board(dimension, num_mines)
    state_counter = 0
    mine_times = []
    game_over = False 
        
    while not game_over and state_counter < dimension**2:
        
        # Find all available locations
        locs = np.where(state == 0)
        places = list(zip(locs[0], locs[1], locs[2]))
        new = [(x,y) for (x,y,z) in places if z == 0]
        if len(new) == 0:
            game_over = True
            
        # Random Choose from list
        next_action_loc = random.choice(new)
        
        if not game_over:
            x,y = next_action_loc
            if state[x][y][1] == 1:
                mine_times.append(state_counter)
            state = get_next_state(state, next_action_loc, flag = False, playing = True)
            state_counter += 1    
            
    # This helps score the no flag, continued play version
    avg_mine_click = np.mean(mine_times)
    return avg_mine_click


def play_one_game_no_training_baseline(dimension, num_mines, q_network):
    """
    Just making a nice function wrapper to make the code look more logical
    
    The q network passed into here would simply be an untrained version
    """
    return play_one_game(dimension, num_mines, q_network)