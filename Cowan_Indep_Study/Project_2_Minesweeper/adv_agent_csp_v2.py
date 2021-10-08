# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:03:31 2021
@author: joshc
"""
import copy, pprint

import base_functions as bf
import basic_agent as ba
import advanced_agent_constraints as adv

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
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    """
    
    made_changes = False
    repeat = True
    unknowns = ba.find_num_unknowns_on_board(covered_board)
    while repeat and unknowns > 0:
        repeat = False
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
                        fact_dict = adv.assume_loc_is_mine((i,j), fact_dict, reference_board)
                    repeat = True
                    made_changes = True
                    break
                    
            # Rule2
            # known mines == num_possible_mines
            # This means that all hidden neighbors are safe
            if len(v[2]) == v[1]:
                if len(v[3]) != 0:
                    # print("Unknowns all safe!")
                    for (i,j) in v[3]:
                        # print(f"\tR{i}C{j} marked Safe!")
                        covered_board[i][j] = reference_board[i][j]
                        fact_dict = adv.assume_loc_is_safe((i,j), fact_dict, reference_board)
                    repeat = True
                    made_changes = True
                    break
                
        if repeat:
            # print("Rechecking board, no random choice yet.")
            unknowns = ba.find_num_unknowns_on_board(covered_board)
            
    return covered_board, fact_dict, made_changes

def assume_a_single_square(covered_board, reference_board, fact_dict):
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
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    Returns
    -------
    covered_board : List of list of chars
        board for a specific attempt at solving, covered in ? for unseen tiles
    changes_made : boolean
        Whether any updates were made to the covered board due to Constraint Satisfaction checks
    fact_dict : dict of (x,y) -> list of [string, int, list of (x,y), list of (x,y), list of (x,y)]
        All neighbor info for a specific board.
        Cell Location -> Cell Status, Cell Possible Mine Neighbors, and then lists containing 
            locations of neighbors whic are known mine, hidden, or safe
    """
    changes_made = False
    repeat = True
    while repeat:
        repeat = False
        freq_unknown_sorted = adv.get_highest_freq_unknown(fact_dict)
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
            
            # print("\nAssuming Mined")
            fact_dict_M = ba.build_fact_dictionary(covered_board, reference_board)
            assume_mine_fd = adv.assume_loc_is_mine(most_freq_unknown, fact_dict_M, reference_board)
            assume_mine_KM, assume_mine_KS = adv.find_changes_due_to_assumption(assume_mine_fd, reference_board)
            # print("Assumed Mine gives mine: ", assume_mine_KM, "\nAssumed Mine gives safe: ", assume_mine_KS)
            
            # print("\nAssuming Safe")
            fact_dict_S = ba.build_fact_dictionary(covered_board, reference_board)
            assume_safe_fd = adv.assume_loc_is_safe(most_freq_unknown, fact_dict_S, reference_board)
            assume_safe_KM, assume_safe_KS = adv.find_changes_due_to_assumption(assume_safe_fd, reference_board)
            # print("Assumed Safe gives mine: ", assume_safe_KM, "\nAssumed Safe gives safe: ", assume_safe_KS)
            
            # Find the intersection of the results of the two assumptions to show a definite safe/mine no matter the assumption
            new_KM = assume_mine_KM.intersection(assume_safe_KM)
            new_KS = assume_mine_KS.intersection(assume_safe_KS)
    
            # If any new locations are found, update the board
            if len(new_KM) != 0 or len(new_KS) != 0:
                # print(f"***New info found by Constraint Satisfaction from {most_freq_unknown}***")
                # print(f"{len(new_KM)} new mines, {len(new_KS)} new safe spots")
                for (x,y) in new_KM:
                    covered_board[x][y] = reference_board[x][y]
                    fact_dict = adv.assume_loc_is_mine((x,y), fact_dict, reference_board)
                for (x,y) in new_KS:
                    covered_board[x][y] = reference_board[x][y]
                    fact_dict = adv.assume_loc_is_safe((x,y), fact_dict, reference_board)
                changes_made = True
                repeat = True
                break

    return covered_board, changes_made, fact_dict

def run_advanced_agent(covered_board, reference_board, num_mines):
    """
    Run the advanced agent on one instance of a board, and record the number of 
        safe mine identifications and total number of random guesses required
        
    Changes from 
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
    # print("\tAdv Agent v2 Time!")
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
        
        # We only make a random guess if we cannot make any new CSP assumptions
        # or basic fact assumptions on the current board state
        while True:
            try:
                fact_dict = ba.build_fact_dictionary(cbA, reference_board)
                cbA, fact_dict, logic_changes_made = apply_logic_to_fact_dict(cbA, reference_board, fact_dict)
                cbA, assume_changes_made, fact_dict = assume_a_single_square(cbA, reference_board, fact_dict)
                if not logic_changes_made and not assume_changes_made:
                    break
            except Exception:
                break
        unknown = ba.find_num_unknowns_on_board(cbA)
            
    # print("-"*5 + "Board Solved"+ "-"*5)
    # print(f"Final Score: {total_score} out of {num_mines} safely found!")
    # print(f"Random Moves Needed: {random_guess}")
    return total_score, random_guess

# covered = [["1", "?", "2"],
#             ["?", "?", "?"],
#             ["?", "3", "?"]]
# reference = [["1", "M", "2"],
#               ["2", "4", "M"],
#               ["M", "3", "M"]]
# bf.print_board(covered)
# # cbA, change = adv.assume_a_single_square(covered, reference)
# tsb, c = ba.run_basic_agent(covered, reference, 4)
# print(tsb, c)
# ts, rg = run_advanced_agent(covered, reference, 4)
# print(ts, rg)