# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 18:47:30 2021

@author: joshc
"""

'''
General Idea:
    
init bandits
choose initial q values
    are we optimistic or zero starting?

for run in numRuns    
    for step in n steps total
        choose action
            for all options, random select if tie
            1) greedy
            2) epsilon greedy 
        update average reward and timestep data
        update action values
            1) regular update, where r(n) is initial q value (qstar)
                q(n+1) = q(n) + alpha(R(n) + q(n))
                
            2) Upper Confidence Bounded
                q(n+1) = q(n) + alpha(sqrt(ln(n))/num times action chosen)
    
    update general time step data with new random run

divide general time step data by numRuns
graph avg reward over time


Data Considerations:
    will be taking n data point a total of numRuns times
    keep single run in df column, and expand df as num runs increases
        to get final output, find avg of row?
        df['mean'] = df.mean(axis=1)
    
    will need to keep sep dataframe for distinct setups, so will init after parameters chosen
    
    can use dict to keep track of number of times action chosen for UCB calc

'''
import random, math
import numpy as np
import pandas as pd

def choose_first_guesses(k, init_guess_type):
    """
    Parameters
    ----------
    k : int
        number of bandits to choose from
    spread : int
        Scaling factor to indicate possible max reward
    guess_type : string
        either 'zero' or 'optimized'

    Returns
    -------
    init_guesses : list of ints
        the first guess of q values for all bandits
    """
    init_guesses = []
    if init_guess_type == 'zero':
        init_guesses = [0 for _ in range(k)]
    elif init_guess_type == 'optimistic':
        init_guesses = [3 for _ in range(k)]
    else:
        print("Please enter either 'zero' or 'optimistic' as your guess type")
    return init_guesses

def choose_action(q_values, epsilon, time_step, action_freq = [], c_value = 0):
    """
    Parameters
    ----------
    q_values : list of floats
        the current best guesses for action values of all bandits
    epsilon : float
        the random guessing parameter, if it is zero all choices are greedy
    action_freq : list of ints
        how many time each action has been chosen, for use in UCB only
    c : float
        how much to value previous information
    
    Returns
    -------
    action_choice : int
        which bandit to choose in the next time step
    """
    # UCB Selection: A(n) = argmax(q(a,n) + c*(sqrt(ln(n))/num times action chosen))
    if action_freq and c_value != 0:
        l = []
        #If there is any action that hasn't been chosen yet, it is considered optimal
        for index, num in enumerate(action_freq):
            if num == 0:
                l.append(index)
        if len(l) != 0:
            return random.choice(l)
        temp_q = [q_values[i] + c_value * math.sqrt(math.log(time_step)/action_freq[i]) for i in range(len(q_values))]
        return np.argmax(temp_q)
        
    # Epsilon-Greedy choice
    elif epsilon != 0:
        rand_choice = random.random()
        
        # Exploration
        if rand_choice <= epsilon:
            return random.randint(0, len(q_values)-1)
        
    # Greedy choice (Exploitation)
    return np.argmax(q_values)
    
def update_q_vals(q_values, alpha, q_star_values, action_choice, time_step):
    # Cheap hack to lower alpha over time
    if alpha < 0:
        alpha = 1/time_step
        
    # Regular Update for single action: q(n+1) = q(n) + alpha(R(n) - q(n))
    q_values[action_choice] = q_values[action_choice] + alpha*(q_star_values[action_choice] - q_values[action_choice])
    
    return q_values

def get_reward_for_action(q_star_values, action_choice):
    q_a = q_star_values[action_choice]
    reward = random.gauss(q_a, 1)
    return reward

def execute_run(time_steps, k, init_guess_type, epsilon, alpha, c_value = 0):
    q_star_values = [random.gauss(0,1) for _ in range(k)]
    # print('*', [round(i, 2) for i in q_star_values])
    q_values = choose_first_guesses(k, init_guess_type)
    total_reward = 0
    avg_reward = {0:0}
    action_freq = []

    # UCB Action Counter Only   
    if c_value != 0:
        action_freq = [0 for _ in range(k)]
    
    # Generate Data from multiple action choices
    for time_step in range(1, time_steps+1):
        
        # if time_step%250 == 0:
        #     print(time_step, [round(i, 2) for i in q_values])
        
        action_choice = choose_action(q_values, epsilon, time_step, action_freq, c_value)
        q_values = update_q_vals(q_values, alpha, q_star_values, action_choice, time_step)
        total_reward += get_reward_for_action(q_star_values, action_choice)
        avg_reward[time_step] = total_reward/time_step
        if c_value != 0:
            action_freq[action_choice] += 1
    
    # Convert to series for future data work
    return pd.Series(avg_reward)    