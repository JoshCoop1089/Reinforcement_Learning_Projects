# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 19:30:28 2021

@author: joshc
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:23:10 2021

@author: joshc
"""

"""

Board layout (State)
    3xNxN
    3 Layers -> Clicked or Flagged, Is it a Mine, Clue
        Valid Actions only if Is clicked = 0
            Is flagged can be -1, clicked can be 1?
        Is Mine layer holds answers to board
        Clue Layer holds number of mine information

            
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
import random, copy
import numpy as np
import tensorflow as tf

def make_board(dimension, num_mines, testing = False):
    """
    Create a dimXdim board with a random assortment of mines
    
    Data Format is a 3xdimXdim np array
    board[x][y][0] -> Clicked/Flagged Layer (0 for unclicked, 1 for clicked, eventually -1 for flag)
    board[x][y][1] -> Mine Layer (0 for no mine, 1 for mine) 
    board[x][y][2] -> Clue Layer (-1 if mine is present, otherwise 1-8  for num of neighbor mines)
                                      (but scaled down to [0.125,1] cause TF no like big nums)
    """
    board = np.zeros((dimension,dimension,3))
    
    # Generate mines and place on board
    mine_locs = set()
    if num_mines > dimension**2: num_mines = dimension**2
    while len(mine_locs) < num_mines:
        i,j = random.randint(0,dimension-1), random.randint(0,dimension-1)
        mine_locs.add((i,j))
    mine_locs = list(mine_locs)
    for (x,y) in mine_locs:
        board[x][y][1] = 1
        board[x][y][2] = -1
        
        
    # Generate clue values
    for i in range(dimension):
        for j in range(dimension):
            
            # Location is a mine
            if (i,j) in mine_locs:
                continue
            
            # Location isn't a mine
            neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                         (i, j-1),             (i, j+1),
                         (i+1, j-1), (i+1, j),(i+1, j+1)]
            for (x,y) in neighbors:
                if 0<=x<dimension and 0<=y<dimension:
                    if board[x][y][1] == 1:
                        board[i][j][2] += 0.125
    if testing:
        return board, mine_locs
    else:
        return board

def print_board(board, full_print = False):
    """
    full_print = False will output the board with only clicked spots
    
    full_print = True will output the whole board, mine locations and clues
    """
    print()
    dimension = len(board[1])
    # Small board labeling of rows and columns for easier visuals
    if dimension <= 10:
        print("\t", end = '')
        for num in range(dimension):
            print(num, end = ' ')
        print("\n\t" + "-"*(2*dimension-1))
    
    for i in range(dimension):
        if dimension <= 10:
            print(str(i) + " |", end = ' ')

        for j in range(dimension):
            # Will need to update this for flag representations
                # correct flag = 'C'
                # incorrect flag = 'F'
            
            # Space isn't clicked
            if board[i][j][0] != 1 and full_print == False:
                print("-", end = ' ')
            else:
                # Space isn't Mined
                if board[i][j][1] != 1:
                    print(int(8*board[i][j][2]), end = ' ')
                else:
                    print('M', end = ' ')
        print()
    print()
   
# # Don't know if i need an incremental reward retrieval but will keep for notes on versions of program
# def reward_from_action(board, location):
#     """
#     Will have to manipulate this function depending on reward structures
#     1st draft:
#         Only clicks, no flags, allow for playing through mine clicks
#     2nd version: (not implemented yet)
#         Only clicks, no flags
#         Allow for playing through mine clicks
#         Allow for mine click being end of game
#     3rd version: (not implemented yet)
#         Clicks and flags
#         Allow for playing through mine clicks
#         Allow for mine clicks being end of game
#         Deal with proper flag placement as a positive reward
#         Improper flag placement is a negative reward    
#     """    
    
#     (row,column) = location
#     reward = 0
    
#     """Version 1: Play through mines"""
#     # Location is a mine
#     if board[1][row][column] == 1:
#         reward = -1
        
#     # Location is not a mine
#     else:
#         reward = 0
        
#     return reward

def make_reward_board(board):
    """
    This will be more complex on future reward function versions
    See detailed notes in reward_from_action function above
    """

    reward_board = np.zeros(board.shape[0:2])
    
    # Reward for mine is -1, otherwise 0
    for x in range(dimension):
        for y in range(dimension):
            reward_board[x][y] = -1 * board[x][y][1]

    
    return reward_board

def q_value_update(state1_q_board, state2_q_board, reward_board, learning_param):
    """
    state1_q_board: holds the q values for a specific state of the board
    state2_q_board: holds the q values for the next state of the board
    reward_board: gives the rewards for every space in the board state
    
    Thoughts:
        Network.predict(input_state) -> gives us the old_q_board
        We use this function to update to new_q_board
        
        We return new_q_board, and store it, and input_state, 
            in the history buffer to batch update the network?
            
        We generate the next state by doing an argmax over new_q, and repeat until end of game?
    
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
        
    # Get list of all available spots
    locs = np.where(board == 0)
    places = list(zip(locs[0], locs[1], locs[2]))
    new = [(x,y) for (x,y,z) in places if z == 0]
    if len(new) == 0:
        game_over = True
        
    # Get location of highest q in available locations
    else:
        q_list = [q_board[0][x][y] for (x,y) in new]
        # print(q_list)
        next_action_loc = new[np.argmax(q_list)]
        # print(next_action_loc)
        
    return next_action_loc, action_is_flag, game_over

def get_next_state(board_c, location, flag = False):
    board = copy.deepcopy(board_c) 
    x,y = location
    if not flag:
        board[x][y][0] = 1
    else:
        board[x][y][0] = -1
    return board

def update_network_from_one_episode(dimension, num_mines, learning_param, batch_fraction, q_network):
    """
    Run through a single game starting with a new board.
    
    Choose batch_percent of the state transitions to use to update the q_network
    """
    
    history = {}
    
    state_t1 = make_board(dimension, num_mines)
    state_counter = 0
    reward_board = make_reward_board(state_t1)
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
    q_network.fit(states, labels)
    return q_network
    

# """
# How do you start off the network?
# Network first guess at layers:
#     Conv2d with input shape as 3xdimxdim
    
#     some middle layers?
    
#     Final layer:
#         Dense, with dimxdim nodes
#         Reshape layer, turns dense into a (dim,dim), since dense turns out a 1d vector

# """

dimension = 10
num_mines = 25
learning_param = 0.1
batch_fraction = 4
input_shape = (dimension, dimension, 3)


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



def init_data(dimension, num_mines):
    state = [make_board(dimension, num_mines) for x in range(dimension**2)]
    label = [np.zeros((dimension, dimension)) for x in range(dimension**2)]
    states = tf.convert_to_tensor(state)
    labels = tf.convert_to_tensor(label)
    return states, labels
    

states, labels = init_data(dimension, num_mines)
q_network.fit(states, labels)

for _ in range(3):
    q_network = update_network_from_one_episode(dimension, num_mines, learning_param, batch_fraction, q_network)



# board, mine_locs = make_board(dimension, num_mines, testing = True)
# print("Full Board w/ mines")
# print_board(board, full_print = True)
# rewards = make_reward_board(board)

# s1 = np.zeros(board.shape[0:2])
# s2 = np.zeros(board.shape[0:2])
# new_q = q_value_update(s1, s2, rewards, learning_param)

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
# print_board(board, full_print = False)

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