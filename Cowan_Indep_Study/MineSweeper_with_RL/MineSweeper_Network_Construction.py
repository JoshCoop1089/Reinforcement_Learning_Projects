# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 14:15:29 2021

@author: joshc
"""
import MineSweeper_Base_Game as ms
import MineSweeper_TF_Functions_Regular_Q as rq
import MineSweeper_Replay_Buffer_Enhanced as rbe

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
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
    label = [np.random.rand(dimension, dimension) for x in range(dimension**2)]
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
def create_base_network(filters, kernel_size, input_variables, network_variables):
        
    dimension, num_mines, learning_param, batch_fraction, _ = input_variables
    dropout_coef, l2_val, dense_layer_nodes = network_variables
    input_shape = (dimension, dimension, 3)

    q_network = tf.keras.Sequential([
                    layers.Conv2D(filters = filters[0], kernel_size = kernel_size[0], activation='relu', input_shape = input_shape),
                    layers.Conv2D(filters = filters[1], kernel_size = kernel_size[1], activation='relu'),
                    layers.Flatten(),
                    layers.Dropout(dropout_coef),
                    layers.Dense(dense_layer_nodes[0], activation = 'relu', kernel_regularizer=regularizers.l2(l2_val)),
                    layers.Dropout(dropout_coef),
                    layers.Dense(dense_layer_nodes[1], activation = 'relu', kernel_regularizer=regularizers.l2(l2_val)),
                    layers.Reshape(target_shape=(dimension, dimension))
    ])
    # q_network.summary()
    
    q_network.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['accuracy'])
    
    # states, labels = init_data(dimension, num_mines)
    # q_network.fit(states, labels)
    return q_network

def train_q_network_without_good_buffer(num_training_times, num_episodes_per_update, input_variables, q_network):
    for i in range(num_training_times):
        print(f"\n==> Single Q v1:\n\tTraining Batch #{i+1} out of {num_training_times} <==")
        q_network = rq.update_network_from_multiple_episodes(input_variables, q_network, num_episodes_per_update)
    return q_network

def train_q_network_with_good_buffer(input_variables, buffer_variables, num_training_times, q_networks):
    update_type = "RegularQ"
    for i in range(num_training_times):
        print(f"\n==> Single Q v2:\n\tTraining Batch #{i+1} out of {num_training_times} <==")
        q_networks = rbe.update_network_with_tiered_buffer(input_variables, buffer_variables, q_networks, update_type)
    return q_networks[0]

# def train_double_q_networks(input_variables, buffer_variables, num_training_times, q_networks):
#     update_type = "RegularQ"
#     for i in range(num_training_times):
#         print(f"\n==> Training Batch #{i+1} out of {num_training_times} <==")
#         q_networks[0] = rbe.update_network_with_tiered_buffer(input_variables, buffer_variables, q_networks, update_type)
#     return q_networks[0]

# def train_q_network_with_good_buffer(input_variables, buffer_variables, num_training_times, q_networks):
#     update_type = "RegularQ"
#     for i in range(num_training_times):
#         print(f"\n==> Training Batch #{i+1} out of {num_training_times} <==")
#         q_networks[0] = rbe.update_network_with_tiered_buffer(input_variables, buffer_variables, q_networks, update_type)
#     return q_networks[0]






