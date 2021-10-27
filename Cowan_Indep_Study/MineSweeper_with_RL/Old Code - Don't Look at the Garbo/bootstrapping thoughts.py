# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 18:52:45 2021

@author: joshc
"""

"""
Best guess at mine network style:
    
    Single dense layer with 80 * dim^2 layer
    
    dropout coef of 0.44
    
Thoughts about generating training data from mine predictions to use in q learning training.

Q learning basic workflow:
    State 1 -> Network -> Q_Board_1
    
    argmax(Q_Board_1) -> next location
    
    state 1 + next loc  -> state 2
    
    state 2 -> network -> q_board_2
    
    q_board 1 updated from q_board2 -> updated q_board_1
    
    store state 1 and updated q board 1
    
    state 2 becomes state 1, and repeat


Does using the mine network as the trainer insert it as the selector of the next location only?

ie, instead of argmax(q_board_1) you throw 
    state 1 -> mine network -> argmin(predictions) -> next location
    then use that to get state 2, and then get q_board2, and continue as before?

"""

# Define Board Parameters

# Define Network Parameters

# Define Training Parameters

# def train_single_network_reducing_epsilon(input_variables, network_name, epsilon_steps):
    
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

import MineSweeper_Base_Game as ms
from Minesweeper_Non_Perfect_Info import one_hot_encode_next_state

def generate_test_data(dimension, num_mines, num_games):
    state_counter = 0
    history = {}
    for _ in range(num_games):
        board = ms.make_board(dimension, num_mines)
        mine_board = board[:,:,1]
        state = np.zeros((dimension, dimension, 11))
        game_over = False
        
        while not game_over:
            # Generate random next board state
            # Find all available locations
            locs = np.where(board[:,:,0] == 0)
            places = list(zip(locs[0], locs[1]))
            
            if len(places) == 0:
                game_over = True
            else:
                history[state_counter] = (state, mine_board)
                # Random Choose from list
                next_loc = random.choice(places) 
                state = one_hot_encode_next_state(state, board, next_loc)  
                state_counter += 1
    return history

def freq_dict(inputs):
    freq_dic = {}
    for val in inputs:
        freq_dic[val] = freq_dic.get(val,0)+1
    return freq_dic

def distinct_vals_b_minus_a(a, b):
    a_freq = freq_dict(a)
    b_freq = freq_dict(b)
    abkeys = set([*a_freq, *b_freq])
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

def get_network_play_results (num_games, dimension, num_mines, network_name):
    avg_score = []    
    avg_times = []
    vals = [x*num_games//10 for x in range (11)]
    for x in range(1,num_games+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games}")
        board = ms.make_board(dimension, num_mines)
        b_enc = np.zeros(input_shape)
        count = 0
        mine_times = []
        
        while count < dimension ** 2:
            b_enc_t = tf.convert_to_tensor(np.expand_dims(b_enc, axis=0))
            pred = network_name.predict(b_enc_t)
            locs = np.where(board[:,:,0] == 0)
            places = list(zip(locs[0], locs[1]))
            actions = [pred[0][x][y] for (x,y) in places]
        
            next_loc = places[np.argmin(actions)]
            if board[next_loc[0]][next_loc[1]][1] == 1:
                mine_times.append(count)
            b_enc = one_hot_encode_next_state(b_enc, board, next_loc)
            count += 1
            
        avg_times.append(np.mean(mine_times))

    avg_score = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_times]
    return avg_score 

dimension = 3    
mine_percent = 0.4

num_games_train = 1000
num_training_rounds = 75
num_games = 500

dropout = 0.375
dense_size = 60*(dimension**2)

num_mines = int(mine_percent * (dimension**2))
input_shape = (dimension, dimension, 11)

model = tf.keras.Sequential([
                tf.keras.Input(shape = input_shape),
                layers.Flatten(),
                layers.Dropout(dropout),
                layers.Dense(dense_size),
                layers.Dropout(dropout),
                layers.Dense(dimension**2, activation='sigmoid'),
                layers.Reshape(target_shape=(dimension, dimension))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy']) 

checkpoint_filepath = '\model_checkpoints\checkpoint'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                    filepath=checkpoint_filepath,
                                    save_weights_only=True,
                                    monitor='accuracy',
                                    mode='max',
                                    save_best_only=True)

loss = []
accuracy = []
for _ in range(num_training_rounds):    
    state = []
    label = []
    history = generate_test_data(dimension, num_mines, num_games_train)  
    batch = random.sample(history.items(), 20*(dimension**2))
    
    for k, (s,l) in batch:
        state.append(s)
        label.append(l)
    states = tf.convert_to_tensor(state)
    labels = tf.convert_to_tensor(label)
    outputs = model.fit(states, labels, callbacks = [model_checkpoint_callback])
    
    loss.extend(outputs.history['loss'])
    accuracy.extend(outputs.history['accuracy'])

# Training Loss Over Time!
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo')
plt.plot(epochs, accuracy, 'ro')

avg_score_end = get_network_play_results(num_games, dimension, num_mines, model)

# The model weights (that are considered the best) are loaded into the model.
model.load_weights(checkpoint_filepath)
trained_results = get_network_play_results(num_games, dimension, num_mines, model)

fig, ax = plt.subplots(1,1, sharex='col', sharey = 'row', figsize=(10,10)) 
fig1, ax1 = plt.subplots(1,1, sharex='col', sharey = 'row', figsize=(10,10))
fig.suptitle(f"Unique Scores from Random and Trained for {num_games} games\nBoard Size: {dimension}x{dimension} || Mines: {num_mines}" + 
                  "\n"+ "- "*20 + "\nBlue: End of Training, Orange: Best Model")
fig1.suptitle(f"Score Distributions for {num_games} games\nBlue: End of Training, Orange: Best Model" + 
                  f"\n Blue Mean: {round(np.mean(avg_score_end), 1)} || Orange Mean: {round(np.mean(trained_results), 1)}")

# MAKE THE UBERGRAPH UNLEASH THE KRAKEN
display_data = []
display_data.append(avg_score_end)
display_data.append(trained_results)

# Histograms!
a_out, b_out = distinct_vals_b_minus_a(avg_score_end, trained_results)      
ax.bar(a_out.keys(), a_out.values())
ax.bar(b_out.keys(), b_out.values())

# KDE's! But Double! And Vertical! Called Violins! With Quartiles!
sns.violinplot(data = display_data, legend = False, ax = ax1, inner = 'quartile')