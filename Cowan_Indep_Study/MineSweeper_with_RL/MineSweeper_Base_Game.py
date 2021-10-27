# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:23:10 2021

@author: joshc

Board layout (State)
    NxNx3
    3 Layers -> Clicked or Flagged, Is it a Mine, Clue
        Valid Actions only if Is clicked = 0
            Is flagged can be -1, clicked can be 1?
        Is Mine layer holds answers to board
        Clue Layer holds number of mine information
"""
import random
import numpy as np

def make_board(dimension, num_mines, testing = False):
    """
    Create a dimXdim board with a random assortment of mines
    
    Data Format is a dimXdimx3 np array
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
    dimension = board.shape[0]
    # Reward for mine is -1, otherwise 0
    for x in range(dimension):
        for y in range(dimension):
            reward_board[x][y] = -1 * board[x][y][1]
    return reward_board

def optimal_play_percent(dimension, num_mines, score):
    """
    Game Type: Play through to the end
    Avg number of safe space clicked before mine?
        How to deal with the end of the game when there are minimal safe spaces?
        
    *** Find the mean time when all mines have been clicked
            Keep track of which move clicks on a mine
            End of the game, find the average turn
            The higher the value, the more mines were clicked later in the game, 
                ie, the network made safe choices earlier, and was only left with mines later
    """

    safe_spaces = dimension**2 - num_mines
    
    low = np.mean([x for x in range(0,num_mines)])
    high = np.mean([x for x in range(safe_spaces, dimension**2)])
    
    percentile = (score - low)/(high - low)
    return percentile

def avg_time_per_game(avg, new_game_time, game_num):
    new_avg = (avg * (game_num - 1) + new_game_time)/game_num
    return round(new_avg, 3)