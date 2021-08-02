# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:25:55 2021

@author: joshc
"""
import pandas as pd
from operator import itemgetter


# 1) (5 points) Let C(x) be the minimal expected cost of being in cell x. 
    # What is C(S) (in terms of its neighbors, cell (1; 0) and (0; 1))?
# 2) (5 points) What is C((1; 0)) in terms of its neighbors?
# 3) (15 points) For any cell x, give a mathematical expression for C(x). 
    # Be clear on what each term or factor represents.
# 4) (10 points) If you had the value of C(x) for each cell, how could you use 
    # it to find the best direction to move in cell S?
# 5) (30 points) Compute C(x) for each x. What is C(S)?
# 6) (10 points) Determine for each cell what the best direction to move in is. 
    # Show this on the grid.

def print_grid(grid):
    df = pd.DataFrame(grid)
    print(df)
    
def id_neighbors(grid, location):
    (i,j) = location
    poss_neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    neighbors = []
    for (x,y) in poss_neighbors:
        if 0<=x<len(grid) and 0<=y<len(grid[0]):
            neighbors.append((x,y))
    return neighbors

def get_orthogonal_moves(grid, location, next_loc):
    (cur_x, cur_y) = location
    (next_x, next_y) = next_loc
    
    # Orthogonal Slips are column shifts
    if cur_y == next_y:
        orthag = [(cur_x, cur_y-1), (cur_x, cur_y+1)]
        
    # Orthogonal Slips are row shifts
    else:
        orthag = [(cur_x-1, cur_y), (cur_x+1, cur_y)]
        
    valid_ortho = []
    for (x,y) in orthag:
        if 0<=x<len(grid) and 0<= y <len(grid[0]):
            valid_ortho.append((x,y))
    return valid_ortho

def get_cost_of_loc(grid, cost_grid, location, ravine_cost):
    neighbors = id_neighbors(grid, location)
    cost = []
    for poss_move in neighbors:
        poss_cost = -1
        ortho_moves = get_orthogonal_moves(grid, location, poss_move)
        valid = len(ortho_moves)
        
        # Cost of moving in direction chosen
        (next_x, next_y) = poss_move
        loc_cost = cost_grid[next_x][next_y]
        poss_cost += (1-(valid/10))*loc_cost
        
        # Cost of slipping in orthogonal directions
        for (x,y) in ortho_moves:
            ortho_cost = cost_grid[x][y]
            poss_cost += 0.1*ortho_cost
            
        cost.append((poss_move, poss_cost))
    
    max_cost = max(cost,key=itemgetter(1))
    # print(f"Location: {location}")
    # print(cost)
    # print(f"Returning: {max_cost}\n")
    return max_cost

def make_temp_grids (width, length, ravines, ravine_cost):
    grid_temp = [['.' for j in range(width)] for i in range(length)]
    cost_grid_temp = [[5.0 for j in range(width)] for i in range(length)]  
    for i in range(length):
        for j in range(width):
            if (i,j) in ravines:
                grid_temp[i][j] = "hole"
                cost_grid_temp[i][j] = ravine_cost
    grid_temp[length-1][0] = "goal!"
    cost_grid_temp[length-1][0] = 0.00
    return grid_temp, cost_grid_temp

def dist_to_goal(i,j):
    (loc_x,loc_y) = (16,0)
    manhat_dist = abs(loc_x-i) + abs(loc_y-j)
    if j == 0 and i < 15:
        manhat_dist += 2
    return -1*manhat_dist

def run_value_iteration(width, length, ravine_cost):
    
    # Not that it matters, but making the inital guess the shortest path length seemed nice
    cost_grid = [[dist_to_goal(i,j) for j in range(width)] for i in range(length)]
    grid = [['.' for j in range(width)] for i in range(length)]
    ravines = [(i,0) for i in range(2,7)]
    ravines.extend([(i,width-1) for i in range(6,11)])
    ravines.extend([(i,0) for i in range(10,15)])
    for i in range(length):
        for j in range(width):
            if (i,j) in ravines:
                grid[i][j] = 'hole'
                cost_grid[i][j] = ravine_cost
    cost_grid[length-1][0] = 0.0
    
    # Value Iteration until Costs converge (Convergence happens after 68 iterations)
    for i in range(100):  
        grid_temp, cost_grid_temp = make_temp_grids(width, length, ravines, ravine_cost)
        for i in range(length):
            for j in range(width):
                
                # Only updating costs on squares which arent goal or ravine
                if cost_grid[i][j] != 0 and grid[i][j] != "hole":
                    max_cost = get_cost_of_loc(grid, cost_grid, (i,j), ravine_cost)
                    grid_temp[i][j] = max_cost[0]
                    cost_grid_temp[i][j] = max_cost[1]
                    # if (i,j) == (2,1):
                        # print(f"*** Update on loc: {(i,j)} ***")
                        # print(f"Cost used to be: {cost_grid[i][j]}\t|next action: {grid[i][j]}")
                        # print(f"Updates, cost: {cost_grid_temp[i][j]}\t|next action: {grid_temp[i][j]}")
       
        grid = grid_temp
        cost_grid = cost_grid_temp
        
    # print_grid(grid)
    
    # Find the "optimal" path based on the action chosen per square
        # This path technically ignores slip chance if you travel along it,
        # but represents the best options of each square linked together cohesively
    loc = (0,0)
    path = []
    death = False
    while loc != (length-1, 0) and loc not in ravines:
        path.append(loc)
        loc = grid[loc[0]][loc[1]]
        # print(loc)
        if loc in ravines:
            print(">>>Dive into that ravine kiddo, it's cheaper to die!<<<")
            death = True
            break
        if loc in path:
            print("Forced into infinite loop?")
            print(path)
            break
    path.append(loc)
    
    # Rewrite optimal action in nicer form than (i,j)
    for i in range(length):
        for j in range(width):
            direction = ""
            if cost_grid[i][j] != 0 and (i,j) not in ravines:
                (n_x, n_y) = grid[i][j]
                if n_x > i:
                    direction = "DOWN"
                elif n_x < i:
                    direction = "UP"
                elif n_y > j:
                    direction = "RIGHT"
                elif n_y < j:
                    direction = "LEFT"
                grid[i][j] = direction

    # Final Outputs for states and cost values
    print_grid(grid)
    # print_grid(cost_grid)
    # print(f"Optimal Actions would follow this path: \n\t{path}")
    return death

ravine_cost = -1000.0
width = 5
length = 17
# are_you_dead = run_value_iteration(width, length, ravine_cost)

# # Bonus Question (Integer Guessing)
# for ravine_cost in range(-60, -40, 1):
#     print("\n\t New Ravine Cost!")
#     are_you_dead = run_value_iteration(width, length, ravine_cost)
#     print(f"**For a ravine cost of {ravine_cost}, are you dead in a ditch: {are_you_dead}**")
    
# Integer General Guesses for Ravine Splits
# Ravine Cost greater than -26 --> down, down, Jump in Ravine
# Ravine Cost is -26 --> down, right, down the side of the left ravines
# Ravine Cost is -29, -28, -27 --> right, down, right, then middle to the bottom, then left 
# Ravine Cost less than -29 --> right, right, Run down middle of map, left toward goal when you reach bottom
    

# Code for finding the limiting value of not diving into the ravine (binary search method)
dead_low, dead_high = -28, -24
count = 0
while dead_low < dead_high and count < 50:
    count += 1
    dead_mid = (dead_low + dead_high)/2
    are_you_dead_low = run_value_iteration(width, length, dead_low)
    are_you_dead_mid = run_value_iteration(width, length, dead_mid)
    are_you_dead_high = run_value_iteration(width, length, dead_high)
    if are_you_dead_low == are_you_dead_mid:
        dead_low = dead_mid
    elif are_you_dead_high == are_you_dead_mid:
        dead_high = dead_mid

print(dead_low, dead_high)
print("\n\t New Ravine Cost!")
are_you_dead_low = run_value_iteration(width, length, dead_low)
print(f"**For a ravine cost of {dead_low}, are you dead in a ditch: {are_you_dead_low}**")
print("\n\t New Ravine Cost!")
are_you_dead_high = run_value_iteration(width, length, dead_high)
print(f"**For a ravine cost of {dead_high}, are you dead in a ditch: {are_you_dead_high}**")
# Somewhere between -25.912487219294565 and -25.91248721929456 you stop wanting to jump.
