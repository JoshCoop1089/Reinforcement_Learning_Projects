# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 11:11:44 2021

@author: joshc
"""
import random, pprint
import base_functions as bf


def uncover_random_spot(covered_board, reference_board):
    """
    When you cannot make any move based on inference, or just want to 
        flip a tile to see what happens

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving

    Returns
    -------
    covered_board : List of List of chars
        board for a specific attempt at solving, covered in ? for unseen tiles, with one less ?
    
    """
    is_covered = True
    while is_covered:
        i,j = random.randint(0,dimension-1), random.randint(0,dimension-1)
        if covered_board[i][j] == "?":
            is_covered = False
            covered_board[i][j] = reference_board[i][j]
    return covered_board

def uncover_specific_spot(covered_board, reference_board, cell_info, x = -1, y = -1):
    """
    Used to uncover a specific spot.  
    X/Y inputs are -1 by default to allow for user choice, can be input as 
        values if AI Agent is making choice.

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving
    cell_info : dict of (x,y) -> int
        Contains current info about nearby number of mines
    x : int, optional
        X location of spot to uncover, AI makes choice is value is passed in. The default is -1.
    y : int, optional
        Y location of spot to uncover, AI makes choice is value is passed in. The default is -1.

    Returns
    -------
    covered_board : List of List of chars
        board for a specific attempt at solving, covered in ? for unseen tiles, with one less ?
    
    """
    valid_spot = False
    # Player choice, input via command line
    if x == -1 and y == -1:
        while not valid_spot:
            location = str(input("--Manual User Choice--\nWhat spot would you like to uncover?" +\
                                 "\nInput Format: row,column (no parentheses, 0 indexed)" +\
                                 "\nExample Input:\t 1,2\nInput:  "))
            x,y = int(location.split(",")[0]), int(location.split(",")[1])
            try:
                if covered_board[x][y] != "?":
                    print("Spot already uncovered, please input again.")
                else:
                    valid_spot = True
            except Exception as e:
                print("Spot created an error, please try again.")
                print("Exception: ", e)
    print()
            
    # Location choice is assumed to be valid, either by AI or by forced Player choice
    covered_board[x][y] = cell_info[(x,y)]
    return covered_board

"""
Thoughts on how to represent a fact uncovered by a cell
    Tuple (int, list of locations)
    The int holds how many mines need to be present
    and the list of locations is the possible places where those mines could be
    
Can set up rule when int == len(loc_list) all loc's are safely marked as mines

How to deal with inference and creating new rules?
Intersection of lists?

"""


def return_possible_mines_equation(covered_board, reference_board,knowledge_base, i, j):
    """
    Given a location i,j in the board, this will generate the equation pertaining
        to possible mine locations around that specific location.  It will take 
        into account whether neighbors have been explored, and what the total 
        known mine count is for the specific square

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving
    i : int, optional
        row location of spot to search around
    j : int, optional
        column location of spot to search around

    Returns
    -------
    output_string : string
        Semi equation of valid unknown neighbors/mine count, or indication that chosen location is a mine
        
    knowledge_base : dict(k,v): int -> list of (x,y)
        creates a fact about a revealed square, where the int is the number of possible mines, 
        and the list contains all the possible locations for a given square

    """
    valid_unknown_neighbors = []
    known_mines = 0
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                         (i, j-1),             (i, j+1),
                         (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(reference_board) and 0<=y<len(reference_board):
            if covered_board[x][y] == "?":
                valid_unknown_neighbors.append((x,y))
            elif covered_board[x][y] == "M":
                known_mines += 1
         
    output_string = f"R:{i} C:{j} give the following new information:\n"
    for num, (x,y) in enumerate(valid_unknown_neighbors):
        output_string += str((x,y)) 
        if num != len(valid_unknown_neighbors)-1:
            output_string += " + "
            
    num_neighbors = reference_board[i][j]
    # If loc is a mine, give no new information
    if num_neighbors == "M":
        output_string = f"R:{i} C:{j} is a mine"
        
    # Otherwise, take into account how many mines are known, the value of 
    # unknown mine neighbors for the loc, and return new information about neighbors
    else:
        num_neighbors -= known_mines
        output_string += f" = {num_neighbors}"
        knowledge_base[(i,j)] = (num_neighbors, valid_unknown_neighbors)
    
    return output_string, knowledge_base


dimension = 5
num_mines = 15       
covered_board, reference_board, mine_locs, cell_info = bf.make_board(dimension, num_mines)
bf.print_board(covered_board)
bf.print_board(reference_board)

print("-"*dimension*2 + "\n")

# covered_board = uncover_random_spot(covered_board, reference_board, cell_info)
# print_board(covered_board)
i = 4
j = 3
knowledge_base = {}

# for (i,j) in [(2,3),(3,3),(3,2), (4,2)]:
#     covered_board = uncover_specific_spot(covered_board, reference_board, cell_info, i, j)
#     bf.print_board(covered_board)
    
#     sent, knowledge_base = return_possible_mines_equation(covered_board, reference_board, knowledge_base, i, j)
#     pprint.pprint(knowledge_base)


