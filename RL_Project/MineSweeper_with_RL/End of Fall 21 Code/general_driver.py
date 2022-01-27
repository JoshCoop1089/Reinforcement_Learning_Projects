# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 20:45:06 2021

@author: joshc
"""

import Actor_Critic_Network_Full_Epsilon as acfe
import Actor_Critic_Network_Update_On_Every_Epsilon as acue
import Use_Mine_Predictions_As_Training_Limited_Epsilon as mpat
import lose_on_first_mine as lofe

# Board Specifics
dimension = 10   
mine_percent = 0.4

# ############################
# ## Testing Output Numbers ##

# # Q Network Specifics 
# dense_size = 80*(dimension**2)
# dropout_coef = 0.375
# learning_param = 0.1
# num_games_per_q_training = 20
# actor_critic_training_times = 2
# min_delta = 1e-4

# # Evaluation
# num_games_eval = 20
# #############################


# #############################
# ## Actual Data Run Numbers ##

# # Q Network Specifics 
# dense_size = 80*(dimension**2)
# dropout_coef = 0.375
# learning_param = 0.1
# num_games_per_q_training = 500
# actor_critic_training_times = 4
# min_delta = 1e-4

# # Evaluation
# num_games_eval = 3000
# #############################

# q_network_params = (dimension, mine_percent, dense_size, dropout_coef)
# training_params = (learning_param, num_games_per_q_training, actor_critic_training_times, min_delta)

# Use AC and update Actor network every time you increase epsilon value
# acue.run_actor_critic_update_after_every_epsilon_change(q_network_params, training_params, num_games_eval)

# # Use AC and update Actor network after a full 0-100 epsilon run
# acfe.run_actor_critic_update_after_all_epsilon(q_network_params, training_params, num_games_eval)

####################################################################
# Using Mine Predictor Network to Feed single Q network

# ############################
# ## Testing Output Numbers ##

# # Mine Predictor Specifics 
# num_games_mine_train = 20
# num_training_rounds = 2
# mine_dropout_coef = 0.44
# mine_dense_size = 80*(dimension**2)

# # Q Network Specifics 
# dense_size = 80*(dimension**2)
# dropout_coef = 0.375
# learning_param = 0.1
# num_games_per_epsilon_level_training = 50
# min_delta = 5e-5

# # Evaluation
# num_games_eval = 50
# #############################


#############################
## Actual Data Run Numbers ##

# Mine Predictor Specifics 
num_games_mine_train = 2000
num_training_rounds = 10
mine_dropout_coef = 0.44
mine_dense_size = 80*(dimension**2)

# Q Network Specifics 
dense_size = 80*(dimension**2)
dropout_coef = 0.375
q_learning_param = 0.1
num_games_per_epsilon_level_training = 1000
min_delta = 5e-5

# Evaluation
num_games_eval = 2000
#############################

q_network_params = (dimension, mine_percent, dense_size, dropout_coef)
q_training_params = (q_learning_param, num_games_per_epsilon_level_training, min_delta)
mine_training_params = (num_games_mine_train, num_training_rounds, mine_dropout_coef, mine_dense_size)

mpat.mine_predictor_fed_q_learning_limited_epsilon_runs(q_network_params, q_training_params, mine_training_params, num_games_eval)
# lofe.mine_predictor_fed_q_learning_limited_epsilon_runs_lose_on_first_mine(q_network_params, q_training_params, mine_training_params, num_games_eval)
