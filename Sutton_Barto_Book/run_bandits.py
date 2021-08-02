# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:54:40 2021

@author: joshc
"""
import bandit_functions as bf

import pandas as pd
import matplotlib.pyplot as plt

######################
# General Parameters #
######################

number_of_bandits = 5
time_steps = 1000
number_of_runs = 200


# Specific Bandit Choices
init_guess_type = 'zero'  #Set to 'zero' for no initial guess of q values
epsilon = 0                     # Set to non zero val for non greedy bandit
alpha = 0.1                     # Step size for updating q values
c_value = 0                     # Set to non zero value to use UCB action choice

results = pd.DataFrame()
for i in range(number_of_runs):
    results[f'Run {i}'] = bf.execute_run(time_steps, number_of_bandits, \
                                         init_guess_type, epsilon, alpha, c_value)
# print(results.head())
results["avg"] = results.mean(axis=1)

# Specific Bandit Choices
init_guess_type = 'optimistic'  #Set to 'zero' for no initial guess of q values
epsilon = 0.1                    # Set to non zero val for non greedy bandit
alpha = 0.1                     # Step size for updating q values
c_value = 0                     # Set to non zero value to use UCB action choice
results1 = pd.DataFrame()
for i in range(number_of_runs):
    results1[f'Run {i}'] = bf.execute_run(time_steps, number_of_bandits, \
                                         init_guess_type, epsilon, alpha, c_value)
# print(results.head())
results1["avg"] = results1.mean(axis=1)

# Specific Bandit Choices
init_guess_type = 'optimistic'  #Set to 'zero' for no initial guess of q values
epsilon = 0                     # Set to non zero val for non greedy bandit
alpha = 0.1                     # Step size for updating q values
c_value = 2                     # Set to non zero value to use UCB action choice
results2 = pd.DataFrame()
for i in range(number_of_runs):
    results2[f'Run {i}'] = bf.execute_run(time_steps, number_of_bandits, \
                                         init_guess_type, epsilon, alpha, c_value)
# print(results.head())
results2["avg"] = results2.mean(axis=1)


fig, ax1 = plt.subplots()
ax1.plot(results['avg'], color = 'g', label = "Opti, E = 0, A = 0.1, C = 0")
ax1.plot(results1['avg'], color = 'r', label = "Opti, E = 0.1, A = 0.1, C = 0")
ax1.plot(results2['avg'], color = 'b', label = "Opti, E = 0, A = 0.1, C = 2")
ax1.legend()