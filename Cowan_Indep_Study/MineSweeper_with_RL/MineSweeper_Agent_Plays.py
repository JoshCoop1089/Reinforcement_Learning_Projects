# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 17:41:40 2021

@author: joshc
"""
import random
import tensorflow as tf
import numpy as np

import MineSweeper_Base_Game as ms
from MineSweeper_TF_Functions_Regular_Q import get_greedy_next_action
from MineSweeper_TF_Functions_Regular_Q import get_next_state
from MineSweeper_Replay_Buffer_Enhanced import double_q_greedy_next_action

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

def play_one_game_single_network(dimension, num_mines, q_network):
    
    state = ms.make_board(dimension, num_mines)
    state_counter = 0
    mine_times = []
    game_over = False 
        
    while not game_over and state_counter < dimension**2:
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        state_q_board = q_network.predict(state_tensor)
        next_action_loc, _ = get_greedy_next_action(state_q_board, state, epsilon = 0)
        if state_counter < dimension ** 2:
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

def play_one_game_double_q(dimension, num_mines, q_networks):
    
    state = ms.make_board(dimension, num_mines)
    state_counter = 0
    mine_times = []
    game_over = False 
    network_one, network_two = q_networks
        
    while not game_over and state_counter < dimension**2:
        state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0))
        state_q_board_one = network_one.predict(state_tensor)
        state_q_board_two = network_two.predict(state_tensor)
        next_action_loc, _ = double_q_greedy_next_action(state_q_board_one, state_q_board_two, state, epsilon = 0)
        if state_counter < dimension ** 2:
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