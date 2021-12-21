# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 20:03:51 2021

@author: joshc
"""
import random

def make_board(dimension, num_mines):
    """
    Create two boards, one a fully covered version with ? to indicate an 
        unexplored square, and the other a fully visible board with all squares uncovered.
        The second will act as a reference board to compare with the first board, 
        which we allow the agent to slowly uncover

    Parameters
    ----------
    dimension : int
        size of the board
    num_mines : int
        how many mines are placed on the board

    Returns
    -------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving
    mine_locs : list of (x,y) tuples
        specific locations of all placed mines
    cell_info : dict of (x,y) -> ints 
        storage container for all info about a specific cell in the board.
        Currently only holds neighbor mine number info
    """
    if num_mines >= dimension**2: num_mines = dimension**2
    covered_board = [["?" for j in range(dimension)] for i in range(dimension)]
    reference_board = [["?" for j in range(dimension)] for i in range(dimension)]
    
    # Generate mines and place on board
    mine_locs = set()
    while len(mine_locs) < num_mines:
        i,j = random.randint(0,dimension-1), random.randint(0,dimension-1)
        mine_locs.add((i,j))
    mine_locs = list(mine_locs)
    for (x,y) in mine_locs:
        reference_board[x][y] = 'M'
    
    # Identify number of mines relative to each square
    reference_board = identify_number_of_nearby_mines(reference_board, mine_locs)
                
    return covered_board, reference_board, mine_locs

def identify_number_of_nearby_mines(reference_board, mine_locs):
    """
    Given a reference board, fill in the count of nearby mines for each cell
    
    Parameters
    ----------
    reference_board : List of list of chars
        reference board for a specific attempt at solving, contains mine
        locations directly, but when put into this function doesn't 
        contain neighbor mine counts yet
    mine_locs : list of (x,y) tuples
        specific locations of all placed mines
        
    Returns
    -------
    reference_board : List of list of chars
        reference board for a specific attempt at solving, contains mine
        locations and neighbor mine counts
    """
    # Will need to make some kind of mine holder general thing to collapse all info and not cause type problems
    cell_info = {}
    for i in range(len(reference_board)):
        for j in range(len(reference_board)):
            neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                         (i, j-1),             (i, j+1),
                         (i+1, j-1), (i+1, j),(i+1, j+1)]
            cell_info[(i,j)] = 0
            for (x,y) in neighbors:
                if 0<=x<len(reference_board) and 0<=y<len(reference_board):
                    if reference_board[x][y] == "M":
                        cell_info[(i,j)] += 1
                        
    for i in range(len(reference_board)):
        for j in range(len(reference_board)):
            if reference_board[i][j] != "M":
                reference_board[i][j] = str(cell_info[(i,j)])
                
    return reference_board

def print_board_with_column_nums(board):
    ### This will break a bit if the dim is over 10, 
    ### I'm just making it to test out manual spot choices
    print("\t", end = '')
    for num in range(len(board)):
        print(num, end = ' ')
    print("\n\t" + "-"*(2*len(board)-1))
    for i in range(len(board)):
        print(str(i) + " |", end = ' ')
        for j in range(len(board)):
            print(board[i][j], end = " ")
        print()
    print()

def print_board_plain(board):
    #For when boards are bigger than 10x10
    for i in range(len(board)):
        for j in range(len(board)):
            print(board[i][j], end = " ")
        print()
    print()

def print_board(board):
    #Makes life simpler when we need to look at a big board for time/space 
    # complexity analysis vs a small board for manual choice trials 
    # to check our implication logic step by step
    if len(board) <= 10:
        print_board_with_column_nums(board)
    else:
        print_board_plain(board)

# dimension = 10
# num_mines = 60       
# covered_board, reference_board, mine_locs = make_board(dimension, num_mines)
# print_board(covered_board)
# print_board(reference_board)

# dimension = 12
# num_mines = 40        
# covered_board, reference_board, mine_locs, cell_info = make_board(dimension, num_mines)
# print_board(covered_board)
# print_board(reference_board)
