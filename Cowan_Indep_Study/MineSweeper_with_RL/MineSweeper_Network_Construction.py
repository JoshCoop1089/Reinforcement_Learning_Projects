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

"""
First Attempt at Network Structure 
    What does it do, why does it do it?  
    What layer choices are needed
    what is the filter size doing for us
    change kernel size of conv2d?
    do we need multiple conv2d layers and some maxpool like in the tutorials?
    
"""

def train_network(filters, kernel_size, num_training_times, input_variables):
    
    dimension, num_mines, learning_param, batch_fraction, num_episodes_per_update = input_variables
    input_shape = (dimension, dimension, 3)

    q_network = tf.keras.Sequential([
                    tf.keras.layers.Conv2D(filters = filters, kernel_size = kernel_size, input_shape = input_shape),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(dimension*dimension),
                    tf.keras.layers.Reshape(target_shape=(dimension, dimension))
    ])
    q_network.summary()
    
    q_network.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    states, labels = init_data(dimension, num_mines)
    q_network.fit(states, labels)
    
    for _ in range(num_training_times):
        q_network = mstf.update_network_from_multiple_episodes(input_variables, q_network, num_episodes_per_update)
        
    return q_network