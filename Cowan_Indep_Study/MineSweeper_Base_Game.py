# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:23:10 2021

@author: joshc
"""

"""

Board layout (State)
    NxNx3
    3 Layers -> Clicked or Flagged, Is it a Mine, Clue
        Valid Actions only if Is clicked = 0
            Is flagged can be -1, clicked can be 1?
        Is Mine layer holds answers to board
        Clue Layer holds number of mine information

            
Network Ideas:
    Input:
        One board state (NxNx3 tensor)
    Output:
        Q-value of all viable actions from input
            (NxNx1 tensor, with a masking over any spot already clicked)
            input[x][y][0] == 1 would indicate that (x,y) would need a masking on output
        *** When flags are needed, output would become NxNx2, to account for click or flag Q values)
    
    
    
"""
import random
import numpy as np

def make_board(dimension, num_mines):
    """
    Create a dimXdim board, and with a random assortment of mines
    
    Data Format is a dimXdimX3 np array
    board[x][y][0] -> Clicked/Flagged Layer (0 for unclicked, 1 for clicked, eventually -1 for flag)
    board[x][y][1] -> Mine Layer (0 for no mine, 1 for mine) 
    board[x][y][2] -> Clue Layer (-1 if mine is present, otherwise 1-8 for num of neighbor mines)
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
    for i in range(len(board)):
        for j in range(len(board)):
            
            # Location is a mine
            if (i,j) in mine_locs:
                continue
            
            # Location isn't a mine
            neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                         (i, j-1),             (i, j+1),
                         (i+1, j-1), (i+1, j),(i+1, j+1)]
            for (x,y) in neighbors:
                if 0<=x<len(board) and 0<=y<len(board):
                    if board[x][y][1] == 1:
                        board[i][j][2] += 0.125
    
    return board

def print_board(board, full_print = False):
    """
    full_print = False will output the board with only clicked spots
    
    full_print = True will output the whole board, mine locations and clues
    """
    
    # Small board labeling of rows and columns for easier visuals
    if len(board) <= 10:
        print("\t", end = '')
        for num in range(len(board)):
            print(num, end = ' ')
        print("\n\t" + "-"*(2*len(board)-1))
    
    for i in range(len(board)):
        if len(board) <= 10:
            print(str(i) + " |", end = ' ')

        for j in range(len(board)):
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
    
def reward_from_action(board, location):
    """
    Will have to manipulate this function depending on reward structures
    1st draft:
        Only clicks, no flags, allow for playing through mine clicks
    2nd version: (not implemented yet)
        Only clicks, no flags
        Allow for playing through mine clicks
        Allow for mine click being end of game
    3rd version: (not implemented yet)
        Clicks and flags
        Allow for playing through mine clicks
        Allow for mine clicks being end of game
        Deal with proper flag placement as a positive reward
        Improper flag placement is a negative reward    
    """    
    
    (row,column) = location
    reward = 0
    
    """Version 1: Play through mines"""
    # Location is a mine
    if board[row][column][1] == 1:
        reward = -1
        
    # Location is not a mine
    else:
        reward = 0
        
    return reward


        

dimension = 10
num_mines = 40
board = make_board(dimension, num_mines)
print_board(board, full_print = True)

board[3][4][0] = 1
print_board(board, full_print = False)
