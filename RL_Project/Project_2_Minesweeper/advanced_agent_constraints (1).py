# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 10:52:45 2021

@author: joshc
"""

"""
Advanced Agent using Basic Logic and Constraint Satisfaction:
1) Create frequency table for all spots in the hidden neighbors list
2) Select the square which appears the most
3) Prepare two copies of the KB, KM, and KS lists
4)
    a) Add this square to the known mines list
    b) Update the KB with this assumption, and see if any facts produce additional changes to the KS/KM lists
5)
    a) Add this square to the known safe list
    b) Update the KB with this assumption, and see if any facts produce additional changes to the KS/KM lists
6) Compare any KS/KM additions beside the step2 additions
    6a) If any square shows up on both KM/KS, then update the original KB/KS/KM lists
"""
import pprint, copy

import basic_agent as ba
import base_functions as bf

def get_highest_freq_unknown(fact_dict):
    """
    Create an ordered dictionary of how many equations all unknown squares occur
        in the fact_dictionary, to use as guide for choosing which square to 
        assume in the start of the CSP

    Parameters
    ----------
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe 

    Returns
    -------
    freq_sorted : dict of (x,y) -> int
        How many different times a specific x,y location shows up as an unknown 
        in the current fact dictionary

    """
    freq_dict = {}
    
    # V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
    for neighbor_info in fact_dict.values():
        for loc in neighbor_info[3]:
            freq_dict[loc] = freq_dict.get(loc,0)+1

    freq_sorted = {k:v for k,v in sorted(freq_dict.items(), key = lambda x: x[1], reverse = True)}
    # print(freq_sorted)
    return freq_sorted


def assume_loc_is_safe(loc, fact_dict, reference_board):
    """
    Given a specific location, change a copy of the basic fact dictionary to 
        assume said location is actually safe, not unknown

    Parameters
    ----------
    loc : tuple of (x,y)
        Row x, Column y spot to be changed from unknown to safe
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    reference_board : List of list of chars
        reference board for a specific attempt at solving

    Returns
    -------
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe

    """
    i,j = loc[0], loc[1]
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                          (i, j-1),             (i, j+1),
                          (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(reference_board) and 0<=y<len(reference_board):
            # V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
            try:
                fact_dict[(x,y)][3].remove(loc)
                fact_dict[(x,y)][4].append(loc)
                # print(f"Changing {loc} to safe in {(x,y)}")
            except Exception:
                continue
            
    return fact_dict

def assume_loc_is_mine(loc, fact_dict, reference_board):
    """
    Given a specific location, change a copy of the basic fact dictionary to 
        assume said location is actually a mine, not unknown

    Parameters
    ----------
    loc : tuple of (x,y)
        Row x, Column y spot to be changed from unknown to mine
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    reference_board : List of list of chars
        reference board for a specific attempt at solving

    Returns
    -------
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    """
    i,j = loc[0], loc[1]
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                          (i, j-1),             (i, j+1),
                          (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(reference_board) and 0<=y<len(reference_board):
            # V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
            try:
                fact_dict[(x,y)][3].remove(loc)
                fact_dict[(x,y)][2].append(loc)
                # print(f"Changing {loc} to mine in {(x,y)}")
            except Exception:
                continue
            
    return fact_dict

def find_changes_due_to_assumption(fact_dict, reference_board):
    """
    Given a fact dictionary, this will extrapolate any assumptions that can be 
        made, and then return the set of locations which have been found due 
        to basic agent logic.  This is essentially basic agent on steroids 
        without the additional choice of random location choice at the end

    Parameters
    ----------
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    reference_board : List of list of chars
        reference board for a specific attempt at solving

    Returns
    -------
    new_known_mines : set of (x,y) tuples
        All mine locations that can be assumed based on the initial fact_dictionary
    new_known_safe : set of (x,y) tuples
        All safe locations that can be assumed based on the initial fact_dictionary

    """
    new_known_mines = set()
    new_known_safe = set()
    update_needed = True
    while update_needed:
        repeat = False
        
        # Iterate through the dictionary, checking each fact for possible updates
        # V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
        for (key,v) in fact_dict.items():
            # print(key, v)

            # If you find that an update is possible, immediately update 
            # the fact_dict, and start again with the new dict
            
            # Rule1
            # Num_possible_mines - known mines == hidden_neighbors
            #  This means that all hidden neighbors are mines
            if v[1] - len(v[2]) == len(v[3]):
                if len(v[3]) != 0:
                    # print(f"Looking at {key}: -> {v}")
                    # print("Unknowns all mined!")
                    for new_mine in v[3]:
                        new_known_mines.add(new_mine)
                        fact_dict = assume_loc_is_mine(new_mine, fact_dict, reference_board)
                    repeat = True
                    break
                    
            # Rule2
            # known mines == num_possible_mines
            # This means that all hidden neighbors are safe
            if len(v[2]) == v[1]:
                if len(v[3]) != 0:
                    # print(f"Looking at {key}: -> {v}")
                    # print("Unknowns all safe!")
                    for new_safe in v[3]:
                        new_known_safe.add(new_safe) 
                        fact_dict = assume_loc_is_safe(new_safe, fact_dict, reference_board)
                    repeat = True
                    break
        if not repeat:
            update_needed = False
        
    return new_known_mines, new_known_safe


def assume_a_single_square(covered_board, reference_board):
    """
    Given a covered board, generate a list of all locations which are the most 
        frequently occuring unknown squares (frequently appearing in other 
        locations facts)  Then go through this list of unknowns one by one, 
        making the dual assumptions of "Safe" or "Mined", and see what effect 
        the initial assumption has on the rest of the board.  If you ever get 
        a spot that is safe/mined from both assumptions, update the covered board,
        and repeat the procedure until there are no new tiles to change

    Parameters
    ----------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    reference_board : List of list of chars
        reference board for a specific attempt at solving

    Returns
    -------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    changes_made : boolean
        Whether any updates were made to the covered board due to COnstrain Satisfaction checks
    """
    changes_made = False
    fac_dict = ba.build_fact_dictionary(covered_board, reference_board)
    freq_unknown_sorted = get_highest_freq_unknown(fac_dict)
    assumption_squares = []
    
    # Find all unknown locations which occur in at least 2 facts
    for (key, val) in freq_unknown_sorted.items():
        if val > 1:
            # print(key, val)
            assumption_squares.append(key)
    
    # For each frequent unknown, make the assumption of "safe/mines" and see 
    # what happens if you percolate that assumption through the board using only the basic agent logic
    for most_freq_unknown in assumption_squares:
        # print("\n--> Changing value for: ", most_freq_unknown)
        if covered_board[most_freq_unknown[0]][most_freq_unknown[1]] != "?":
            continue
        fac_dict = ba.build_fact_dictionary(covered_board, reference_board)
        
        # print("\nAssuming Mined")
        # pprint.pprint(fact_dict)
        fact_dict = copy.deepcopy(fac_dict)
        assume_mine_fd = assume_loc_is_mine(most_freq_unknown, fact_dict, reference_board)
        assume_mine_KM, assume_mine_KS = find_changes_due_to_assumption(assume_mine_fd, reference_board)
        # print("Assumed Mine gives mine: ", assume_mine_KM, "\nAssumed Mine gives safe: ", assume_mine_KS)
        
        # print("\nAssuming Safe")
        # pprint.pprint(fact_dict)
        fact_dict = copy.deepcopy(fac_dict)
        assume_safe_fd = assume_loc_is_safe(most_freq_unknown, fact_dict, reference_board)
        assume_safe_KM, assume_safe_KS = find_changes_due_to_assumption(assume_safe_fd, reference_board)
        # print("Assumed Safe gives mine: ", assume_safe_KM, "\nAssumed Safe gives safe: ", assume_safe_KS)
        
        # Find the intersection of the results of the two assumptions to show a definite safe/mine no matter the assumption
        new_KM = assume_mine_KM.intersection(assume_safe_KM)
        new_KS = assume_mine_KS.intersection(assume_safe_KS)

        # Error checking the supposed new mined/safe locations (a precaution from early iterations due to some incorrect choices)
        # total_right_mines = 0
        # incorrect = 0
        # for (x,y) in list(new_KM):
        #     if reference_board[x][y] == "M":
        #         total_right_mines += 1
        #     else:
        #         incorrect += 1
        #         print(f"Not Mine: {(x,y)}")
        # print("\nNew Known Mines: ", new_KM, "\nCorrect Predictions: ", total_right_mines, "\tIncorrect: ", incorrect)
        
        # total_right_safe = 0
        # incorrect = 0
        # for (x,y) in list(new_KS):
        #     if reference_board[x][y] != "M":
        #         total_right_safe += 1
        #     else:
        #         incorrect += 1
        #         print(f"Not Safe: {(x,y)}")
        # print("New Known Safes: ", new_KS, "\nCorrect Predictions: ", total_right_safe, "\tIncorrect: ", incorrect)
        
        # If any new locations are found, update the board
        if len(new_KM) != 0 or len(new_KS) != 0:
            changes_made = True
            # print(f"***New info found by Constraint Satisfaction from {most_freq_unknown}***")
            # print(f"{len(new_KM)} new mines, {len(new_KS)} new safe spots")
            for (x,y) in new_KM:
                val = reference_board[x][y]
                if val == "M":
                    covered_board[x][y] = val
                # else:
                #     print(f"Incorrect Mine Predicted at {(x,y)}")
            for (x,y) in new_KS:
                val = reference_board[x][y]
                if val != "M":
                    covered_board[x][y] = val
                # else:
                #     print(f"Incorrect Safe Predicted at {(x,y)}")
        
    return covered_board, changes_made

def run_advanced_agent(covered_board, reference_board, num_mines):
    """
    Run the advanced agent on one instance of a board, and record the number of 
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
    random_guess : int
        The number of random guesses required to keep the solver logic moving forward
    """
    # print("\tAdv Agent Time!")
    total_score = num_mines
    random_guess = 0
    unknown = ba.find_num_unknowns_on_board(covered_board)
    cbA = copy.deepcopy(covered_board)
    while unknown > 0:
        # print(f"\n--> Random Guess: Round {random_guess}")
        cbA, mine_detonated = ba.uncover_random_spot(cbA, reference_board)
        random_guess += 1
        if mine_detonated:
            total_score -= 1
        # bf.print_board(cbA)
        fact_dict = ba.build_fact_dictionary(cbA, reference_board)
        # pprint.pprint(fact_dict)
        cbA = ba.apply_logic_to_fact_dict(cbA, reference_board, fact_dict)
        
        # We only make a random guess if we cannot make any new CSP assumptions
        # or basic fact assumptions on the current board state
        while True:
            try:
                # print("Adv tring Inferences")
                cbA, changes_made = assume_a_single_square(cbA, reference_board)
                if not changes_made:
                    break
                fact_dict = ba.build_fact_dictionary(cbA, reference_board)
                # pprint.pprint(fact_dict)
                cbA = ba.apply_logic_to_fact_dict(cbA, reference_board, fact_dict)
            except Exception:
                break       
        # bf.print_board(cbA)
        unknown = ba.find_num_unknowns_on_board(cbA)
            
    print("-"*5 + "Board Sovled"+ "-"*5)
    print(f"Final Score C: {total_score} out of {num_mines} safely found!")
    print(f"Random Moves Needed C: {random_guess}")
    return total_score, random_guess

# dimension = 5
# num_mines = 10
# covered_board, reference_board, mine_locs = bf.make_board(dimension, num_mines)
# score, count = run_advanced_agent(covered_board, reference_board, num_mines)
# print("final score: ", score, "\n total guesses", count)
