# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 18:51:40 2021

@author: joshc
"""
import time
import matplotlib.pyplot as plt

import base_functions as bf
import basic_agent as ba
import advanced_agent_constraints as adv
import adv_agent_csp_v2 as adv2
import advanced_agent_equations as adv3

# Board Parameters
dimension = 10
attempts = 20
num_intervals = 20

# General Graphing Code
percent_chunk = (dimension**2)//num_intervals
mine_num_list = [percent_chunk * i for i in range(num_intervals+1)]
mine_percents = [x/(dimension**2) for x in mine_num_list]

mine_scores_basic = []
random_moves_basic = []
times_basic = []

# mine_scores_adv = []
# random_moves_adv = []
# times_adv = []

mine_scores_adv2 = []
random_moves_adv2 = []
times_adv2 = []

mine_scores_adv3 = []
random_moves_adv3 = []
times_adv3 = []

start_time = time.time()
for num_mines in mine_num_list:
    print(f"\n---> Starting: {num_mines} Mines on {dimension}x{dimension} board <---")
    print("---> " + str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))))
    agg_score_basic, agg_rounds_basic, agg_time_basic = 0, 0, 0
    # agg_score_adv, agg_rounds_adv, agg_time_adv = 0, 0, 0
    agg_score_adv2, agg_rounds_adv2, agg_time_adv2 = 0, 0, 0
    agg_score_adv3, agg_rounds_adv3, agg_time_adv3 = 0, 0, 0
    
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
        
        # # Using Adv Agent (v1 CSP w/o Optimizations)
        # adv_start = time.time()
        # total_score_adv, random_guess_adv = adv.run_advanced_agent(cover, rb, num_mines)
        # adv_end = time.time()
        # agg_time_adv += (adv_end-adv_start)
        # agg_score_adv += total_score_adv
        # agg_rounds_adv += random_guess_adv
        
        # Using Adv Agent (v2 CSP with Optimizations)
        adv2_start = time.time()
        total_score_adv2, random_guess_adv2 = adv2.run_advanced_agent(cover, rb, num_mines)
        adv2_end = time.time()
        agg_time_adv2 += (adv2_end-adv2_start)
        agg_score_adv2 += total_score_adv2
        agg_rounds_adv2 += random_guess_adv2
        
        # # Using Adv Agent (v2 CSP with Optimizations)
        adv3_start = time.time()
        total_score_adv3, random_guess_adv3 = adv3.run_advanced_equations(cover, rb, num_mines)
        adv3_end = time.time()
        agg_time_adv3 += (adv3_end-adv3_start)
        agg_score_adv3 += total_score_adv3
        agg_rounds_adv3 += random_guess_adv3

    if num_mines == 0 and agg_score_basic == 0:
        mine_scores_basic.append(1)
        if agg_score_adv3 == 0:
            # mine_scores_adv.append(1)
            mine_scores_adv2.append(1)
            mine_scores_adv3.append(1)
    else:
        mine_scores_basic.append(agg_score_basic/(attempts*num_mines))
        # mine_scores_adv.append(agg_score_adv/(attempts*num_mines))
        mine_scores_adv2.append(agg_score_adv2/(attempts*num_mines))
        mine_scores_adv3.append(agg_score_adv3/(attempts*num_mines))
        
    random_moves_basic.append(agg_rounds_basic/attempts)
    # random_moves_adv.append(agg_rounds_adv/attempts)
    random_moves_adv2.append(agg_rounds_adv2/attempts)
    random_moves_adv3.append(agg_rounds_adv3/attempts)
    
    times_basic.append(agg_time_basic/attempts)
    # times_adv.append(agg_time_adv/attempts)
    times_adv2.append(agg_time_adv2/attempts)
    times_adv3.append(agg_time_adv3/attempts)
    
end_time = time.time()
print(f"\n---> Total time for {attempts*num_intervals} boards: {end_time-start_time:.2f} <---")

fig, ax1 = plt.subplots()
plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n different agents with varying mine percentages")
ax1.set_xlabel('Mines as Percent of Total Board')
ax1.set_ylabel('%Mines Safely Found')
ax1.plot(mine_percents, mine_scores_basic, color = 'g', label = "Basic")
# ax1.plot(mine_percents, mine_scores_adv, color = 'r', label = "Advanced v1")
ax1.plot(mine_percents, mine_scores_adv2, color = 'r', label = "Advanced v2")
ax1.plot(mine_percents, mine_scores_adv3, color = 'b', label = "Advanced v3")
ax1.legend()

fig,ax2 = plt.subplots()
plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n different agents with varying mine percentages")
ax2.set_xlabel('Mines as Percent of Total Board')
ax2.set_ylabel('Random Moves Required')
ax2.plot(mine_percents, random_moves_basic, color = "g", label = "Basic")
# ax2.plot(mine_percents, random_moves_adv, color = "r", label = "Advanced v1")
ax2.plot(mine_percents, random_moves_adv2, color = "r", label = "Advanced v2")
ax2.plot(mine_percents, random_moves_adv3, color = "b", label = "Advanced v3")
ax2.legend()

fig,ax3 = plt.subplots()
plt.title(f"{attempts} tries at solving {dimension}x{dimension} Minesweeper using \n different agents with varying mine percentages")
ax3.set_xlabel('Mines as Percent of Total Board')
ax3.set_ylabel('Avg Solving Time per Board (seconds)')
ax3.plot(mine_percents, times_basic, color = "g", label = "Basic")
# ax3.plot(mine_percents, times_adv, color = "r", label = "Advanced v1")
ax3.plot(mine_percents, times_adv2, color = "r", label = "Advanced v2")
ax3.plot(mine_percents, times_adv3, color = "b", label = "Advanced v3")
ax3.legend()

plt.show()