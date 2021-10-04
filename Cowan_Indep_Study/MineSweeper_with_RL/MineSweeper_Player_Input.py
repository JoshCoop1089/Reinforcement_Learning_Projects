# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 18:49:11 2021

@author: joshc
"""

# Human plays minesweeper for the datas

# build minesweeper board
# human input x,y coord
# print board
# repeat

import MineSweeper_Base_Game as ms
import numpy as np


dimension = 4
num_mines = 8

board = ms.make_board(dimension, num_mines)
count = 0
mine_list = []
while count < dimension**2:
    try:
        x = int(input("Row: "))
        y = int(input("Column: "))
    except Exception as e:
        continue
    if board[x][y][0] == 1:
        continue
    else:
        board[x][y][0] = 1
        if board[x][y][1] == 1:
            mine_list.append(count)
        ms.print_board(board)
        count += 1
    
print("Mean Score: ", ms.optimal_play_percent(dimension, num_mines, np.mean(mine_list)))
