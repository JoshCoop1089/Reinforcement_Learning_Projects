# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:32:56 2021

@author: joshc
"""

"""
Rules for the Basic Agent:
    
Implement the following simple agent as a baseline strategy to compare against your own:

Cell Info:

    whether or not it is a mine or safe (or currently covered)
    if safe, the number of mines surrounding it indicated by the clue
    the number of safe squares identied around it
    the number of mines identied around it.
    the number of hidden squares around it.

Logic Rules:
    
If, for a given cell, the total number of mines (the clue) minus the 
    number of revealed mines is the number of hidden neighbors, every
    hidden neighbor is a mine.
    
If, for a given cell, the total number of safe neighbors (8 - clue) 
    minus the number of revealed safe neighbors is the number of hidden
    neighbors, every hidden neighbor is safe.

If a cell is identied as safe, reveal it and update your information.

If a cell is identied as a mine, mark it and update your information.

The above steps can be repeated until no more hidden cells can be 
    conclusively identied.

If no hidden cell can be conclusively identied as a mine or safe, pick 
    a cell to reveal uniformly at random from the remaining cells.

"""
import random, copy
import pprint
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
    cell_info : dict of (x,y) -> int
        Contains current info about nearby number of mines

    Returns
    -------
    covered_board : List of List of chars
        board for a specific attempt at solving, covered in ? for unseen tiles, with one less ?
    mine_detonated : Boolean
        if the uncovered tile contained a mine
    """
    dim = len(covered_board)
    is_covered = True
    mine_detonated = False
    while is_covered:
        i,j = random.randint(0,dim-1), random.randint(0,dim-1)
        if covered_board[i][j] == "?":
            is_covered = False
            # print(f"\nUncovering R{i}C{j}")
            covered_board[i][j] = reference_board[i][j]
            if reference_board[i][j] == "M":
                # print("KABOOM!")
                mine_detonated = True
    return covered_board, mine_detonated

def get_indiv_cell_info(covered_board, i, j):
    """
    Find all relevant neighbor info for a specific RiCj cell

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    i : int
        row number for chosen tile
    j : int
        DESCRIPTION.

    Returns
    -------
    indiv_cell : list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific cell.
        Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe 
    """
    indiv_cell = []
    status = covered_board[i][j]
    
    # If cell chosen is a mine, isnumeric = False
    if status.isnumeric():
        num_possible_neighbor_mines = int(status)
    else:
        num_possible_neighbor_mines = -1

    known_neighbor_mines = []
    hidden_neighbors = []
    safe_neighbors = []
    
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                         (i, j-1),             (i, j+1),
                         (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(covered_board) and 0<=y<len(covered_board):
            if covered_board[x][y] == "?":
                hidden_neighbors.append((x,y))
            elif covered_board[x][y] == "M":
                known_neighbor_mines.append((x,y))
            elif covered_board[x][y].isnumeric():
                safe_neighbors.append((x,y))
    
    indiv_cell.append(status)
    indiv_cell.append(num_possible_neighbor_mines)
    indiv_cell.append(known_neighbor_mines)
    indiv_cell.append(hidden_neighbors)
    indiv_cell.append(safe_neighbors)
    
    return indiv_cell

def num_unknown_neighbors(i,j, covered_board):
    """
    To limit the creation of facts to only spots which have unknown squares near them

    Parameters
    ----------
    i : int
        Row value of spot to id neighbors of
    j : int
        Column value of spot to id neighbors of
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles

    Returns
    -------
    unknown_neighbors : int
        number of unknown neighbors around a specific spot

    """
    unknown_neighbors = 0
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                     (i, j-1),             (i, j+1),
                     (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(covered_board) and 0<=y<len(covered_board):
            if covered_board[x][y] == "?":
                unknown_neighbors += 1
                
    return unknown_neighbors

def build_fact_dictionary(covered_board, reference_board):
    """
    Creates the basic facts for each uncovered cell on the board.

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving

    Returns
    -------
    cell_info : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe 

    """
    cell_info = {}
    for i in range(len(covered_board)):
        for j in range(len(covered_board)):
            if covered_board[i][j].isnumeric() and num_unknown_neighbors(i,j, covered_board) > 0:
                cell_info[(i,j)] = get_indiv_cell_info(covered_board, i, j)
    return cell_info

def find_num_unknowns_on_board(covered_board):
    """
    Given a board, this counts how close the board is to being solved

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles

    Returns
    -------
    num_unknowns : int
        number of ? tiles left on covered board

    """
    num_unknowns = 0
    for i in range(len(covered_board)):
        for j in range(len(covered_board)):
            if covered_board[i][j] == "?":
                num_unknowns += 1
    return num_unknowns

def apply_logic_to_fact_dict(covered_board, reference_board, fact_dict):
    """
    Given a board state, fill in as many tiles as can be found using single 
        fact inferences.  When you run out, return to main function to randomly
        choose a new tile to check

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe 

    Returns
    -------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    """
    
    made_changes = True
    unknowns = find_num_unknowns_on_board(covered_board)
    while made_changes and unknowns > 0:
        made_changes = False
        # print(len(fact_dict.values()))
        # pprint.pprint(fact_dict)
        # bf.print_board(covered_board)
        
        # V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
        for v in fact_dict.values():
            # Rule1
            # Num_possible_mines - known mines == hidden_neighbors
            #  This means that all hidden neighbors are mines
            if v[1] - len(v[2]) == len(v[3]):
                if len(v[3]) != 0:
                    # print("Unknowns all mined!")
                    for (i,j) in v[3]:
                        # print(f"\tR{i}C{j} marked Mine!")
                        covered_board[i][j] = "M"
                        unknowns -= 1
                    made_changes = True
                    
            # Rule2
            # known mines == num_possible_mines
            # This means that all hidden neighbors are safe
            if len(v[2]) == v[1]:
                if len(v[3]) != 0:
                    # print("Unknowns all safe!")
                    for (i,j) in v[3]:
                        # print(f"\tR{i}C{j} marked Safe!")
                        covered_board[i][j] = reference_board[i][j]
                        unknowns -= 1
                    made_changes = True
        if made_changes:
            # print("Rechecking board, no random choice yet.")
            fact_dict = build_fact_dictionary(covered_board, reference_board)
            unknowns = find_num_unknowns_on_board(covered_board)
            
    return covered_board

def run_basic_agent(covered_board, reference_board, num_mines):
    """
    Run the basic agent on one instance of a board, and record the number of 
        safe mine identifications and total number of random guesses required

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving
    num_mines : int
        how many mines are present on the board (only used for scorekeeping, 
             not known to the agent as part of the global info bonus points)

    Returns
    -------
    total_score : int
        The number of mines safely found by logic 
            (total mines - mines detonated by random guessing)
    count : int
        The number of random guesses required to keep the solver logic moving forward
    """
    # print("\tBasic Agent Time!")
    count = 0
    unknowns = find_num_unknowns_on_board(covered_board)
    total_score = num_mines
    cb = copy.deepcopy(covered_board)
    
    # While a piece of the board is covered, apply basic logic until you 
    # run out of options, then flip over a new square and start again
    while unknowns > 0:
        # print(f"\n--> Random Uncovering: Round {count}")
        cb, mine_detonated = uncover_random_spot(cb, reference_board)
        # bf.print_board(cb)
        if mine_detonated:
            total_score -= 1
        fact_dict = build_fact_dictionary(cb, reference_board)
        # pprint.pprint(fact_dict)
        cb = apply_logic_to_fact_dict(cb, reference_board, fact_dict)
        unknowns = find_num_unknowns_on_board(cb)
        count += 1
        
    print("-"*5 + "Board Solved"+ "-"*5)
    print(f"Final Score basic: {total_score} out of {num_mines} safely found!")
    print(f"Random Moves Needed basic : {count}")
    return total_score, count

# dimension = 5
# num_mines = 10
# covered_board, reference_board, mine_locs = bf.make_board(dimension, num_mines)
# score, count = run_basic_agent(covered_board, reference_board, num_mines)
# print("final score: ", score, "\n total guesses", count)

# dimension = 15
# attempts = 20
# """
# Graphing Basic Agent Score vs Mine Density
# mine density = num_mines / (dim**2)
# """
# five_percent = (dimension**2)//20
# mine_num_list = [five_percent * i for i in range(21)]
# mine_scores = []
# random_moves = []
# for num_mines in mine_num_list:
#     agg_score = 0
#     agg_rounds = 0
#     for index in range(attempts):
#         cb, rb, mloc = bf.make_board(dimension, num_mines)                    
#         # bf.print_board(rb)
#         count = 0
#         unknowns = find_num_unknowns_on_board(cb)
#         total_score = num_mines
#         while unknowns > 0:
#             # print(f"\n--> Random Uncovering: Round {count}")
#             cb, mine_detonated = uncover_random_spot(cb, rb)
#             # bf.print_board(cb)
#             if mine_detonated:
#                 total_score -= 1
#             fact_dict = build_fact_dictionary(cb, rb)
#             # pprint.pprint(fact_dict)
#             cb = apply_logic_to_fact_dict(cb, rb, fact_dict)
#             unknowns = find_num_unknowns_on_board(cb)
#             count += 1
#         print()
#         print("-"*5 + "Board Sovled"+ "-"*5)
#         # bf.print_board(cb)
#         print(f"Final Score: {total_score} out of {num_mines} safely found!")
#         print(f"Random Moves Needed: {count}")
#         agg_score += total_score
#         agg_rounds += count
#     if num_mines == 0 and agg_score == 0:
#         mine_scores.append(1)
#     else:
#         mine_scores.append(agg_score/(attempts*num_mines))
#     random_moves.append(agg_rounds/attempts)

# mine_percents = [x/(dimension**2) for x in mine_num_list]
# fig, ax1 = plt.subplots()
# plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n basic agent with varying mine percentages")
# ax1.set_xlabel('Mines as Percent of Total Board')
# ax1.set_ylabel('Average Percent of Mines Safely Found', color = 'b')
# ax1.scatter(mine_percents, mine_scores, marker = "o", color = 'b')
# ax2 = ax1.twinx()
# ax2.set_ylabel('Average Random Moves Required', color = 'r')
# ax2.scatter(mine_percents, random_moves, marker = "P", color = "r")
# fig.tight_layout()
# plt.show()
