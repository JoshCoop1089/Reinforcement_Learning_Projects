# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 12:11:04 2021
@author: joshc
Added contradiction checking to assumption code to allow for early breaking and new assumptions
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 21:03:31 2021
@author: joshc
"""
import copy, pprint, random

import base_functions as bf
import basic_agent as ba
import advanced_agent_constraints as adv

def assume_loc_is_safe(loc, fact_dict, reference_board, globalInfo = False):
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
    if globalInfo:
        fact_dict['global'][3].remove(loc)
        fact_dict['global'][4].append(loc)
    
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                          (i, j-1),             (i, j+1),
                          (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(reference_board) and 0<=y<len(reference_board):
            try:
                fact_dict[(x,y)][3].remove(loc)
                fact_dict[(x,y)][4].append(loc)
            except Exception:
                continue
            
    return fact_dict

def assume_loc_is_mine(loc, fact_dict, reference_board, globalInfo = False):
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
    if globalInfo:
        fact_dict['global'][3].remove(loc)
        fact_dict['global'][2].append(loc)
    possible_neighbors = [(i-1, j-1), (i-1, j),(i-1, j+1),
                          (i, j-1),             (i, j+1),
                          (i+1, j-1), (i+1, j),(i+1, j+1)]
    for (x,y) in possible_neighbors:
        if 0<=x<len(reference_board) and 0<=y<len(reference_board):
            try:
                fact_dict[(x,y)][3].remove(loc)
                fact_dict[(x,y)][2].append(loc)
            except Exception:
                continue
            
    return fact_dict

def build_fact_dictionary(covered_board, reference_board, globalInfo = False):
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
    if globalInfo:
        num_mines = 0
        for i in range(len(covered_board)):
            for j in range(len(covered_board)):
                if reference_board[i][j] == 'M':
                    num_mines += 1
        cell_info['global'] = ['', num_mines, [], [], []]
    for i in range(len(covered_board)):
        for j in range(len(covered_board)):
            if globalInfo:
                if covered_board[i][j] == '?':
                    loc = 3
                elif covered_board[i][j] == 'M':
                    loc = 2
                else:
                    loc = 4
                cell_info['global'][loc].append((i,j))                    
            
            if covered_board[i][j].isnumeric() and ba.num_unknown_neighbors(i,j, covered_board) > 0:
                cell_info[(i,j)] = ba.get_indiv_cell_info(covered_board, i, j)
    return cell_info

def apply_logic_to_fact_dict(covered_board, reference_board, fact_dict, globalInfo = False):
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
        for v in fact_dict.values():
            # Rule1
            # Num_possible_mines - known mines == hidden_neighbors
            #  This means that all hidden neighbors are mines
            if v[1] - len(v[2]) == len(v[3]):
                if len(v[3]) != 0:
                    for (i,j) in v[3]:
                        covered_board[i][j] = "M"
                        fact_dict = assume_loc_is_mine((i,j), fact_dict, reference_board, globalInfo)
                    repeat = True
                    made_changes = True
                    break
                    
            # Rule2
            # known mines == num_possible_mines
            # This means that all hidden neighbors are safe
            if len(v[2]) == v[1]:
                if len(v[3]) != 0:
                    for (i,j) in v[3]:
                        covered_board[i][j] = reference_board[i][j]
                        fact_dict = assume_loc_is_safe((i,j), fact_dict, reference_board, globalInfo)
                    repeat = True
                    made_changes = True
                    break
                
        if repeat:
            # print("Rechecking board, no random choice yet.")
            unknowns = ba.find_num_unknowns_on_board(covered_board)
            
    return covered_board, fact_dict, made_changes

def find_changes_due_to_assumption(fact_dict, reference_board, globalInfo = False):
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
    contradiction_check : boolean
        if an assumption causes a contradicition in a fact in the KB
    """
    new_known_mines = set()
    new_known_safe = set()
    update_needed = True
    while update_needed:
        repeat = False
        
        # Iterate through the dictionary, checking each fact for possible updates
        # V = [status, num_possible_mines, known_mines, hidden_neighbors, safe_neighbors]
        for (key,v) in fact_dict.items():
            # Check for contradictions in num mines vs num hidden
            # Rule 1 Contradiction
            # Too many mines to place in too few spots
            # num mines - known mines > hidden neighbors
            if v[1] - len(v[2]) > len(v[3]):
                return set(), set(), False
            
            # Rule 2 Contradiction
            # Placed more mines than allowed
            # num mines < known mines
            if v[1] < len(v[2]):
                return set(), set(), False

            # If you find that an update is possible, immediately update 
            # the fact_dict, and start again with the new dict
            
            # Rule1
            # Num_possible_mines - known mines == hidden_neighbors
            #  This means that all hidden neighbors are mines
            if v[1] - len(v[2]) == len(v[3]):
                if len(v[3]) != 0:
                    for new_mine in v[3]:
                        new_known_mines.add(new_mine)
                        fact_dict = assume_loc_is_mine(new_mine, fact_dict, reference_board, globalInfo)
                    repeat = True
                    break
                    
            # Rule2
            # known mines == num_possible_mines
            # This means that all hidden neighbors are safe
            if len(v[2]) == v[1]:
                if len(v[3]) != 0:
                    for new_safe in v[3]:
                        new_known_safe.add(new_safe) 
                        fact_dict = assume_loc_is_safe(new_safe, fact_dict, reference_board, globalInfo)
                    repeat = True
                    break
        if not repeat:
            update_needed = False
        
    return new_known_mines, new_known_safe, True

def assume_a_single_square(covered_board, reference_board, fact_dict, globalInfo = False):
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
    if globalInfo: freq_thres = 2 
    else: freq_thres = 1
    while repeat:
        repeat = False
        freq_unknown_sorted = adv.get_highest_freq_unknown(fact_dict)
        assumption_squares = []
        
        # Find all unknown locations which occur in at least 2 facts
        for (key, val) in freq_unknown_sorted.items():
            if val > freq_thres:
                assumption_squares.append(key)
        
        # For each frequent unknown, make the assumption of "safe/mines" and see 
        # what happens if you percolate that assumption through the board using only the basic agent logic
        for most_freq_unknown in assumption_squares:
            if covered_board[most_freq_unknown[0]][most_freq_unknown[1]] != "?":
                continue
            
            # print("\nAssuming Mined")
            fact_dict_M = build_fact_dictionary(covered_board, reference_board, globalInfo)
            assume_mine_fd = assume_loc_is_mine(most_freq_unknown, fact_dict_M, reference_board, globalInfo)
            assume_mine_KM, assume_mine_KS, no_mine_contradiction = find_changes_due_to_assumption(assume_mine_fd, reference_board, globalInfo)
            # print("Assumed Mine gives mine: ", assume_mine_KM, "\nAssumed Mine gives safe: ", assume_mine_KS)
            
            if no_mine_contradiction:
                # print("\nAssuming Safe")
                fact_dict_S = build_fact_dictionary(covered_board, reference_board, globalInfo)
                assume_safe_fd = assume_loc_is_safe(most_freq_unknown, fact_dict_S, reference_board, globalInfo)
                assume_safe_KM, assume_safe_KS, no_safe_contradiction = find_changes_due_to_assumption(assume_safe_fd, reference_board, globalInfo)
                # print("Assumed Safe gives mine: ", assume_safe_KM, "\nAssumed Safe gives safe: ", assume_safe_KS)
            
            if not no_mine_contradiction:
                # print(f"Contradiction: {most_freq_unknown} must be safe")
                new_KM = set()
                new_KS = set([most_freq_unknown])
            elif no_mine_contradiction and not no_safe_contradiction:
                # print(f"Contradiction: {most_freq_unknown} must be mine")
                new_KM = set([most_freq_unknown])
                new_KS = set()
            else:    
                # Find the intersection of the results of the two assumptions to show a definite safe/mine no matter the assumption
                new_KM = assume_mine_KM.intersection(assume_safe_KM)
                new_KS = assume_mine_KS.intersection(assume_safe_KS)
        
        
            # If any new locations are found, update the board
            if len(new_KM) != 0 or len(new_KS) != 0:
                for (x,y) in new_KM:
                    val = reference_board[x][y]
                    if val == "M":
                        covered_board[x][y] = val
                    else:
                        print(f"Incorrect Mine Predicted at {(x,y)}")
                    fact_dict = assume_loc_is_mine((x,y), fact_dict, reference_board, globalInfo)
                    
                for (x,y) in new_KS:
                    val = reference_board[x][y]
                    if val != "M":
                        covered_board[x][y] = val
                    else:
                        print(f"Incorrect Safe Predicted at {(x,y)}")
                    fact_dict = assume_loc_is_safe((x,y), fact_dict, reference_board, globalInfo)
                changes_made = True
                repeat = True
                break

    return covered_board, changes_made, fact_dict

def check_board(cb, rb):
    for i in range(len(cb)):
        for j in range(len(cb)):
            if cb[i][j] != rb[i][j]:
                return False
    return True

def advanced_location_selection(covered_board, reference_board, fact_dict):
    mine_detonated = False
    
    # Build percent chance of mine from fact dict
    unknown_dict = {}
    for fact in fact_dict.values():
        num_mines_left = fact[1] - len(fact[2])
        if len(fact[3]) != 0:
            mine_chance = round(num_mines_left/len(fact[3]), 2)
            for loc in fact[3]:
                try:
                    unknown_dict[loc].append(mine_chance)
                except Exception:
                    unknown_dict[loc] = [mine_chance]
                    
    percent_dict = {}

    # Fill in rest of unknown squares as 50%
    for i in range(len(covered_board)):
        for j in range(len(covered_board)):
            if covered_board[i][j] == '?':
                percent_dict[(i,j)] = 0.5
            
    # Average all percents for neighbor unknowns from part 1
    for key, val in unknown_dict.items():
        percent_dict[key] = round(sum(val)/len(val), 2)
        
    
    # Choose spot with lowest percent
    (i,j) = min(percent_dict, key = percent_dict.get)
    

    # If multiple spots have the same percent, get one with fewest unknown neighbors
    possibles = {}
    for key, val in percent_dict.items():
        if val == percent_dict[(i,j)]:
            possibles[key] = ba.num_unknown_neighbors(key[0], key[1], covered_board)
    
    (i,j) = min(possibles, key = possibles.get)
    
    
    if covered_board[i][j] == "?":
        # print(f"\nUncovering R{i}C{j}")
        covered_board[i][j] = reference_board[i][j]
        if reference_board[i][j] == "M":
            # print("KABOOM!")
            mine_detonated = True
    return covered_board, mine_detonated


def run_advanced_agent(covered_board, reference_board, num_mines, globalInfo = False, advanced = False, rando = False):
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
    # print("\tAdv Agent v3 Time!")
    total_score = num_mines
    random_guess = 0
    unknown = ba.find_num_unknowns_on_board(covered_board)
    cbA = copy.deepcopy(covered_board)
    fact_dict = build_fact_dictionary(cbA, reference_board, globalInfo)
    while unknown > 0:
        if advanced:
            cbA, mine_detonated = advanced_location_selection(cbA, reference_board, fact_dict)
        else:
            cbA, mine_detonated = ba.uncover_random_spot(cbA, reference_board)
        random_guess += 1
        if mine_detonated:
            total_score -= 1
        
        # We only make a random guess if we cannot make any new CSP assumptions
        # or basic fact assumptions on the current board state
        while True:
            try:
                fact_dict = build_fact_dictionary(cbA, reference_board, globalInfo)
                cbA, fact_dict, logic_changes_made = apply_logic_to_fact_dict(cbA, reference_board, fact_dict, globalInfo)
                cbA, assume_changes_made, fact_dict = assume_a_single_square(cbA, reference_board, fact_dict, globalInfo)
                if not logic_changes_made and not assume_changes_made:
                    # bf.print_board(cbA)
                    break
            except Exception:
                break
        unknown = ba.find_num_unknowns_on_board(cbA)
    '''
    print("-"*5 + "Adv3 Board Solved"+ "-"*5)
    print(f"Final Score: {total_score} out of {num_mines} safely found!")
    print(f"Random Moves Needed: {random_guess}")'''
    
    if not check_board(cbA, reference_board):
        print("BOARD INCORRECT")
              
    return total_score, random_guess

# covered = [["?", "?", "?"],
#             ["?", "?", "?"],
#             ["?", "?", "?"]]
# reference = [["1", "M", "2"],
#               ["2", "4", "M"],
#               ["M", "3", "M"]]
# dimension = 5
# num_mines = 10
# covered, reference, mine_locs = bf.make_board(dimension, num_mines)
# bf.print_board(covered)
# # cbA, change = adv.assume_a_single_square(covered, reference)
# # tsb, c = ba.run_basic_agent(covered, reference, 4)
# # print(tsb, c)
# ts, rg = run_advanced_agent(covered, reference, num_mines, False, True)
# print(ts, rg)

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:51:40 2021
@author: joshc
"""
'''
import time
import matplotlib.pyplot as plt'''

# import base_functions as bf
# import basic_agent as ba
# import advanced_agent_constraints as adv
# import adv_agent_csp_v2 as adv2


'''
# Board Parameters
dimension = 10
attempts = 100
num_intervals = 20

# General Graphing Code
percent_chunk = (dimension**2)//num_intervals
mine_num_list = [percent_chunk * i for i in range(num_intervals+1)]
mine_percents = [x/(dimension**2) for x in mine_num_list]

mine_scores_basic = []
random_moves_basic = []
times_basic = []

mine_scores_adv = []
random_moves_adv = []
times_adv = []

mine_scores_adv2 = []
random_moves_adv2 = []
times_adv2 = []

mine_scores_adv3 = []
random_moves_adv3 = []
times_adv3 = []

mine_scores_adv4 = []
random_moves_adv4 = []
times_adv4 = []

start_time = time.time()
for num_mines in mine_num_list:
    print(f"\n---> Starting: {num_mines} Mines on {dimension}x{dimension} board <---")
    print("---> " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    agg_score_basic, agg_rounds_basic, agg_time_basic = 0, 0, 0
    agg_score_adv, agg_rounds_adv, agg_time_adv = 0, 0, 0
    agg_score_adv2, agg_rounds_adv2, agg_time_adv2 = 0, 0, 0
    agg_score_adv3, agg_rounds_adv3, agg_time_adv3 = 0, 0, 0
    agg_score_adv4, agg_rounds_adv4, agg_time_adv4 = 0, 0, 0
    
    for index in range(attempts):
        print(f"{index}, ", end = " ")
        cover, rb, mloc = bf.make_board(dimension, num_mines)                    
        # bf.print_board(rb)
        
        # Using Basic Agent
        basic_start = time.time()
        total_score, count = ba.run_basic_agent(cover, rb, num_mines)
        basic_end = time.time()
        agg_time_basic += (basic_end-basic_start)
        agg_score_basic += total_score
        agg_rounds_basic += count
        
        # # Using Adv Agent (v1 CSP base)
        adv_start = time.time()
        total_score_adv, random_guess_adv = run_advanced_agent(cover, rb, num_mines, False, False, False)
        adv_end = time.time()
        agg_time_adv += (adv_end-adv_start)
        agg_score_adv += total_score_adv
        agg_rounds_adv += random_guess_adv
        
        # Using Adv Agent (v2 CSP with Advanced Selection)
        adv2_start = time.time()
        total_score_adv2, random_guess_adv2 = comb.run_combined_agent_advanced(cover, rb, num_mines, False, False)
        adv2_end = time.time()
        agg_time_adv2 += (adv2_end-adv2_start)
        agg_score_adv2 += total_score_adv2
        agg_rounds_adv2 += random_guess_adv2

        # Using Adv Agent (v3 CSP with Global Info and Advanced Selection)
        adv3_start = time.time()
        total_score_adv3, random_guess_adv3 = run_advanced_agent(cover, rb, num_mines, False, True)
        adv3_end = time.time()
        agg_time_adv3 += (adv3_end-adv3_start)
        agg_score_adv3 += total_score_adv3
        agg_rounds_adv3 += random_guess_adv3

        adv4_start = time.time()
        total_score_adv4, random_guess_adv4 = comb.run_combined_agent(cover, rb, num_mines)
        adv4_end = time.time()
        agg_time_adv4 += (adv4_end-adv4_start)
        agg_score_adv4 += total_score_adv4
        agg_rounds_adv4 += random_guess_adv4

    if num_mines == 0:
        if agg_score_basic == 0:
            mine_scores_basic.append(1)
        if agg_score_adv3 == 0:
            mine_scores_adv.append(1)
            mine_scores_adv2.append(1)
            mine_scores_adv3.append(1)
            mine_scores_adv4.append(1)
    else:
        mine_scores_basic.append(agg_score_basic/(attempts*num_mines))
        mine_scores_adv.append(agg_score_adv/(attempts*num_mines))
        mine_scores_adv2.append(agg_score_adv2/(attempts*num_mines))
        mine_scores_adv3.append(agg_score_adv3/(attempts*num_mines))
        mine_scores_adv4.append(agg_score_adv4/(attempts*num_mines))
        
    random_moves_basic.append(agg_rounds_basic/attempts)
    random_moves_adv.append(agg_rounds_adv/attempts)
    random_moves_adv2.append(agg_rounds_adv2/attempts)
    random_moves_adv3.append(agg_rounds_adv3/attempts)
    random_moves_adv4.append(agg_rounds_adv4/attempts)
    
    times_basic.append(agg_time_basic/attempts)
    times_adv.append(agg_time_adv/attempts)
    times_adv2.append(agg_time_adv2/attempts)
    times_adv3.append(agg_time_adv3/attempts)
    times_adv4.append(agg_time_adv4/attempts)
    
end_time = time.time()
print(f"\n---> Total time for {attempts*num_intervals} boards: {end_time-start_time:.2f} <---")

fig, ax1 = plt.subplots()
plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n different agents with varying mine percentages")
ax1.set_xlabel('Mines as Percent of Total Board')
ax1.set_ylabel('%Mines Safely Found')
ax1.plot(mine_percents, mine_scores_basic, color = 'g', label = "Basic")
ax1.plot(mine_percents, mine_scores_adv, color = 'r', label = "Combined Agent with Sel")
ax1.plot(mine_percents, mine_scores_adv2, color = 'b', label = "Advanced")
ax1.plot(mine_percents, mine_scores_adv3, color = 'c', label = "Advanced with Sel")
ax1.plot(mine_percents, mine_scores_adv4, color = 'y', label = "Combined")
ax1.legend()

fig,ax2 = plt.subplots()
plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n different agents with varying mine percentages")
ax2.set_xlabel('Mines as Percent of Total Board')
ax2.set_ylabel('Random Moves Required')
ax2.plot(mine_percents, random_moves_basic, color = "g", label = "Basic")
ax2.plot(mine_percents, random_moves_adv, color = "r", label = "Combined Agent with Sel")
ax2.plot(mine_percents, random_moves_adv2, color = "b", label = "Advanced")
ax2.plot(mine_percents, random_moves_adv3, color = "c", label = "Advanced with Sel")
ax2.plot(mine_percents, random_moves_adv4, color = "y", label = "Combined")
ax2.legend()

fig,ax3 = plt.subplots()
plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n different agents with varying mine percentages")
ax3.set_xlabel('Mines as Percent of Total Board')
ax3.set_ylabel('Avg Solving Time per Board (seconds)')
ax3.plot(mine_percents, times_basic, color = "g", label = "Basic")
ax3.plot(mine_percents, times_adv, color = "r", label = "Combined Agent with Sel")
ax3.plot(mine_percents, times_adv2, color = "b", label = "Advanced")
ax3.plot(mine_percents, times_adv3, color = "c", label = "Advanced with Sel")
ax3.plot(mine_percents, times_adv4, color = "y", label = "Combined")
ax3.legend()

plt.show()'''