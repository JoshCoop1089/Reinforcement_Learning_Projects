# -*- coding: utf-8 -*-
"""
Created on Tue Jun 29 20:51:23 2021

@author: joshc

grid rewards -1 for off grid, +10 if at loc (0,1), +5 if at loc (0,3), 0 otherwise
(0,1) moves you to (4,1)
(0,3) moves you to (2,3)

Book Solution for v*
22.0 24.4 22.0 19.4 17.5
19.8 22.0 19.8 17.8 16.0
17.8 19.8 17.8 16.0 14.4
16.0 17.8 16.0 14.4 13.0
14.4 16.0 14.4 13.0 11.7
"""
from operator import itemgetter
def get_reward_from_action(location, action):
    """
    location = (r,c) on grid
    actions = u,l,d,r
    """
    (r,c) = location
    reward = 0
    loc_new = (0,0)
    
    if action == 'u':
        loc_new = (r-1, c)
    elif action == 'd':
        loc_new = (r+1,c)
    elif action == 'l':
        loc_new = (r,c-1)
    elif action == 'r':
        loc_new = (r,c+1)
    
    if loc_new == (0,1):
        loc_new = (4,1)
        reward = 10
    elif loc_new == (0,3):
        loc_new = (2,3)
        reward = 5
        
    (x,y) = loc_new
    if not (0 <= x < 5 and 0 <= y <5):
        reward = -1
        loc_new = location
        
    return loc_new, reward

def get_all_costs(location):
    (i,j) = location
    poss_neighbors = [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]
    values = []
    for next_loc, direction in zip(poss_neighbors, ['d', 'u', 'l', 'r']):
        val_of_new_loc = get_reward_from_action(next_loc, direction)
        
        # Equiprobable movements in all directions means p(s',r|s,a) = 0.25 for all states
        values.append((direction, 0.25*val_of_new_loc))
    return max(values, key=itemgetter(1))
    
def get_value_function(location, grid):
    

grid_val = [[0 for i in range(5)] for j in range(5)]
grid_action = [["" for i in range(5)] for j in range(5)]
