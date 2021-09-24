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
    
"""
import random
import numpy as np

def make_board(dimension, num_mines):
    """
    Create a dimXdim board with a random assortment of mines
    
    Data Format is a 3xdimXdim np array
    board[0][x][y] -> Clicked/Flagged Layer (0 for unclicked, 1 for clicked, eventually -1 for flag)
    board[1][x][y] -> Mine Layer (0 for no mine, 1 for mine) 
    board[2][x][y] -> Clue Layer (-1 if mine is present, otherwise 1-8  for num of neighbor mines)
                                      (but scaled down to [0.125,1] cause TF no like big nums)
    """
    board = np.zeros((3,dimension,dimension))
    
    # Generate mines and place on board
    mine_locs = set()
    if num_mines > dimension**2: num_mines = dimension**2
    while len(mine_locs) < num_mines:
        i,j = random.randint(0,dimension-1), random.randint(0,dimension-1)
        mine_locs.add((i,j))
    mine_locs = list(mine_locs)
    for (x,y) in mine_locs:
        board[1][x][y] = 1
        board[2][x][y] = -1
        
        
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
                    if board[1][x][y] == 1:
                        board[2][i][j] += 0.125
    
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
            # Space isn't clicked
            if board[0][i][j] != 1 and full_print == False:
                print("-", end = ' ')
            else:
                # Space isn't Mined
                if board[1][i][j] != 1:
                    print(int(8*board[2][i][j]), end = ' ')
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
    reward_board = np.zeros(board[0].shape)
    
    # Reward for mine is -1, otherwise 0
    reward_board = -1 * board[1]
    
    return reward_board

def q_value_update(old_q_board, reward_board, learning_param):
    """
    old_q_board: holds the q values for a specific state of the board
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
    
    new_q_board = np.zeros(old_q_board.shape)
    
    q_new = reward_board + gamma * np.amax(old_q_board)
    new_q_board = old_q_board + learning_param * (q_new - old_q_board)
    return new_q_board

        

dimension = 5
num_mines = 10
learning_param = 0.1
board = make_board(dimension, num_mines)
print("Full Board w/ mines")
print_board(board, full_print = True)
rewards = make_reward_board(board)

old_q = np.zeros(board[0].shape)
new_q = q_value_update(old_q, rewards, learning_param)

print("Q values for state, but assumed full board reveal")
print(new_q)
# new_q = q_value_update(new_q, rewards, learning_param)
# print(new_q)
# new_q = q_value_update(new_q, rewards, learning_param)
# print(new_q)

print("\nFirst action taken")
board[0][3][4] = 1
print_board(board, full_print = False)

