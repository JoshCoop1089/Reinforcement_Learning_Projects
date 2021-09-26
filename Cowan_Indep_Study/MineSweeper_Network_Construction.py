# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:15:29 2021

@author: joshc
"""
import MineSweeper_Base_Game as ms
import MineSweeper_TF_Functions as mstf

import tensorflow as tf
import numpy as np
"""
How do you start off the network?
Network first guess at layers:
    Conv2d with input shape as 3xdimxdim
    
    some middle layers?
    
    Final layer:
        Dense, with dimxdim nodes
        Reshape layer, turns dense into a (dim,dim), since dense turns out a 1d vector


Network Ideas:
    Input:
        One board state (3xNxN tensor)
    Output:
        Q-value of all viable actions from input
            (1xNxN tensor, with a masking over any spot already clicked)
            input[0][x][y] == 1 would indicate that (x,y) would need a masking on output
        *** When flags are needed, output would become 2xNxN, to account for click or flag Q values)
    
Things to think about:
    How do we use a masking layer in TF?
    What's the most efficient way of storing the log of states and q_boards for batch updates
    
    Do we update the network using a batch pulled from a fully finished board 
        ie the log of all (state -> new_q_board) choices for a specific starting board
        following some greedy policy for choosing next state based on argmax(new_q_board)    
"""
def init_data(dimension, num_mines):
    state = [ms.make_board(dimension, num_mines) for x in range(dimension**2)]
    label = [np.zeros((dimension, dimension)) for x in range(dimension**2)]
    states = tf.convert_to_tensor(state)
    labels = tf.convert_to_tensor(label)
    return states, labels


dimension = 8
num_mines = 20
learning_param = 0.1
batch_fraction = 4

input_variables = (dimension, num_mines, learning_param, batch_fraction)
input_shape = (dimension, dimension, 3)


"""
First Attempt at Network Structure 
    What does it do, why does it do it?  
    What layer choices are needed
    what is the filter size doing for us
    change kernel size of conv2d?
    do we need multiple conv2d layers and some maxpool like in the tutorials?
    
"""

q_network = tf.keras.Sequential([
                tf.keras.layers.Conv2D(filters = 32, kernel_size = (3,3), input_shape = input_shape),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(dimension*dimension),
                tf.keras.layers.Reshape(target_shape=(dimension, dimension))
])
q_network.summary()

q_network.compile(optimizer='adam',
              loss=tf.keras.losses.MeanSquaredError(),
              metrics=['accuracy'])

################### End of network structure setup ##################

################### BEGIN ZE TESTING NAO ##################
states, labels = init_data(dimension, num_mines)
q_network.fit(states, labels)

for _ in range(20):
    q_network = mstf.update_network_from_one_episode(input_variables, q_network)

mstf.play_one_game(dimension, num_mines, q_network)

# board, mine_locs = ms.make_board(dimension, num_mines, testing = True)
# print("Full Board w/ mines")
# ms.print_board(board, full_print = True)
# rewards = ms.make_reward_board(board)

# s1 = np.zeros(board.shape[0:2])
# s2 = np.zeros(board.shape[0:2])
# new_q = mstf.q_value_update(s1, s2, rewards, learning_param)

# print("Q values for state, but assumed full board reveal")
# print(new_q)
# # # new_q = q_value_update(new_q, rewards, learning_param)
# # # print(new_q)
# # # new_q = q_value_update(new_q, rewards, learning_param)
# # # print(new_q)

# # ((x,y), _, _) = get_greedy_next_action(new_q, board)
# # print(x,y)
# # board[x][y][0] = 1

# print("\nFirst action taken")
# board[1][1][0] = 1
# ms.print_board(board, full_print = False)

# # Using open spaces on board, find spot with maxQ
# locs = np.where(board == 0)
# places = list(zip(locs[0], locs[1], locs[2]))
# new = [(x,y) for (x,y,z) in places if z == 0]
# print(new)
# q_list = [new_q[x][y] for (x,y) in new]
# print(q_list)
# (x,y) = new[np.argmax(q_list)]
# print(x,y)

# board[x][y][0] = 1

# # Using open spaces on board, find spot with maxQ
# locs = np.where(board == 0)
# places = list(zip(locs[0], locs[1], locs[2]))
# new = [(x,y) for (x,y,z) in places if z == 0]
# print(new)
# q_list = [new_q[x][y] for (x,y) in new]
# print(q_list)
# (x,y) = new[np.argmax(q_list)]
# print(x,y)